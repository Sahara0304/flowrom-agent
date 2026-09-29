import numpy as np
import math
from typing import Optional, Tuple

class CandidateModel:
    def __init__(self, latent_dim: int, seed: int = 0):
        assert latent_dim == 52, "This model requires latent_dim=52"
        self.latent_dim = latent_dim
        self.seed = seed
        self.reactant_idx: Optional[np.ndarray] = None
        self.product_idx: Optional[np.ndarray] = None
        self.L_D: Optional[np.ndarray] = None
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        self.fitted = False

    def _pca(self, X: np.ndarray, n_components: int) -> np.ndarray:
        X_centered = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return X_centered @ Vt[:n_components].T

    def _kmeans(self, X: np.ndarray, k: int, n_init: int, max_iter: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        best_inertia = float('inf')
        best_labels = None
        N = X.shape[0]
        for _ in range(n_init):
            idx = rng.choice(N, size=k, replace=False)
            centers = X[idx].copy()
            labels = np.zeros(N, dtype=int)
            for _ in range(max_iter):
                dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                new_labels = np.argmin(dists, axis=1)
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels
                for i in range(k):
                    if np.any(labels == i):
                        centers[i] = X[labels == i].mean(axis=0)
            inertia = np.sum((X - centers[labels]) ** 2)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
        return best_labels

    def _compute_split(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        proj = self._pca(Z, n_components=3)
        labels = self._kmeans(proj, k=2, n_init=10, max_iter=10, seed=self.seed)
        reactant_idx = np.where(labels == 0)[0][:26]
        if len(reactant_idx) < 26:
            remaining = 26 - len(reactant_idx)
            product_candidates = np.where(labels == 1)[0]
            reactant_idx = np.concatenate([reactant_idx, product_candidates[:remaining]])
        product_idx = np.setdiff1d(np.arange(52), reactant_idx)
        assert len(reactant_idx) == 26 and len(product_idx) == 26
        return reactant_idx, product_idx

    def _init_params(self, rng: np.random.Generator):
        self.L_D = np.tril(rng.standard_normal((26, 26)) * 0.1)
        self.W1 = rng.standard_normal((32, 26)) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = rng.standard_normal((26, 32)) * 0.1
        self.b2 = np.zeros(26)

    def _assemble_A(self) -> np.ndarray:
        D = self.L_D @ self.L_D.T
        M_C = np.zeros((52, 52))
        M_C[:26, :26] = -D
        M_C[26:, :26] = D
        return np.eye(52) + M_C

    def _forward(self, z: np.ndarray) -> np.ndarray:
        if z.ndim == 1:
            z = z[None, :]
        Z_R = z[:, self.reactant_idx]
        Z_P = z[:, self.product_idx]
        D = self.L_D @ self.L_D.T
        d = Z_R @ D.T
        Az_R = Z_R - d
        Az_P = Z_P + d
        s1 = Z_R @ self.W1.T + self.b1
        h = np.tanh(s1)
        R = h @ self.W2.T + self.b2
        Z_pred_R = Az_R - R
        Z_pred_P = Az_P + R
        Z_pred = np.empty_like(z)
        Z_pred[:, self.reactant_idx] = Z_pred_R
        Z_pred[:, self.product_idx] = Z_pred_P
        return Z_pred

    def _backward(self, Z: np.ndarray, Z_target: np.ndarray) -> dict:
        B = Z.shape[0]
        Z_R = Z[:, self.reactant_idx]
        Z_P = Z[:, self.product_idx]
        D = self.L_D @ self.L_D.T
        d = Z_R @ D.T
        Az_R = Z_R - d
        Az_P = Z_P + d
        s1 = Z_R @ self.W1.T + self.b1
        h = np.tanh(s1)
        R = h @ self.W2.T + self.b2
        Z_pred_R = Az_R - R
        Z_pred_P = Az_P + R

        Z_pred = np.empty_like(Z)
        Z_pred[:, self.reactant_idx] = Z_pred_R
        Z_pred[:, self.product_idx] = Z_pred_P

        delta = Z_pred - Z_target
        grad_Z_pred = delta / B

        grad_g = grad_Z_pred
        grad_R = -grad_g[:, self.reactant_idx] + grad_g[:, self.product_idx]

        grad_b2 = grad_R.sum(axis=0)
        grad_W2 = grad_R.T @ h
        grad_h = grad_R @ self.W2
        grad_s1 = grad_h * (1.0 - h ** 2)
        grad_b1 = grad_s1.sum(axis=0)
        grad_W1 = grad_s1.T @ Z_R

        grad_Az_R = grad_Z_pred[:, self.reactant_idx]
        grad_Az_P = grad_Z_pred[:, self.product_idx]
        grad_d = -grad_Az_R + grad_Az_P
        grad_D = grad_d.T @ Z_R
        grad_L_D_full = (grad_D + grad_D.T) @ self.L_D
        grad_L_D = np.tril(grad_L_D_full)

        return {
            'L_D': grad_L_D,
            'W1': grad_W1,
            'b1': grad_b1,
            'W2': grad_W2,
            'b2': grad_b2
        }

    def _adam_update(self, params: dict, grads: dict, m: dict, v: dict, t: int, lr: float = 1e-3,
                     beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        for key in params:
            g = grads[key]
            m[key] = beta1 * m[key] + (1.0 - beta1) * g
            v[key] = beta2 * v[key] + (1.0 - beta2) * (g ** 2)
            m_hat = m[key] / (1.0 - beta1 ** t)
            v_hat = v[key] / (1.0 - beta2 ** t)
            params[key] = params[key] - lr * m_hat / (np.sqrt(v_hat) + eps)
        if 'L_D' in params:
            params['L_D'] = np.tril(params['L_D'])

    def fit(self, z_train: np.ndarray):
        assert z_train.ndim == 2 and z_train.shape[1] == 52
        N = z_train.shape[0]
        assert N >= 2, "Need at least 2 time steps"

        # Compute split
        self.reactant_idx, self.product_idx = self._compute_split(z_train)

        # Initialize parameters
        rng = np.random.default_rng(self.seed)
        self._init_params(rng)
        params = {'L_D': self.L_D, 'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}

        # Create pairs
        Z_all = z_train[:-1]
        Z_next_all = z_train[1:]

        # Internal validation split
        n_val = max(1, int(0.1 * len(Z_all)))
        Z_train_pairs = Z_all[:-n_val]
        Z_next_train = Z_next_all[:-n_val]
        Z_val_pairs = Z_all[-n_val:]
        Z_next_val = Z_next_all[-n_val:]

        best_val_loss = float('inf')
        best_params = {k: v.copy() for k, v in params.items()}
        patience = 0
        max_epochs = 200
        patience_max = 30
        batch_size = 32
        t_step = 0

        for epoch in range(max_epochs):
            # Shuffle
            perm = np.random.permutation(len(Z_train_pairs))
            Z_train_pairs = Z_train_pairs[perm]
            Z_next_train = Z_next_train[perm]

            # Mini-batch training
            for i in range(0, len(Z_train_pairs), batch_size):
                Z_batch = Z_train_pairs[i:i+batch_size]
                Z_target_batch = Z_next_train[i:i+batch_size]
                if len(Z_batch) == 0:
                    continue
                grads = self._backward(Z_batch, Z_target_batch)
                t_step += 1
                self._adam_update(params, grads, m, v, t_step)

            # Update self attributes
            self.L_D = params['L_D']
            self.W1 = params['W1']
            self.b1 = params['b1']
            self.W2 = params['W2']
            self.b2 = params['b2']

            # Validation loss
            Z_val_pred = self._forward(Z_val_pairs)
            val_loss = np.mean((Z_val_pred - Z_next_val) ** 2)

            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_params = {k: v.copy() for k, v in params.items()}
                patience = 0
            else:
                patience += 1
                if patience >= patience_max:
                    break

        # Restore best params
        self.L_D = best_params['L_D']
        self.W1 = best_params['W1']
        self.b1 = best_params['b1']
        self.W2 = best_params['W2']
        self.b2 = best_params['b2']
        self.fitted = True

    def predict_next(self, z_t: np.ndarray) -> np.ndarray:
        assert self.fitted, "Model must be fitted first"
        if z_t.ndim == 1:
            z_t = z_t[None, :]
            single = True
        else:
            single = False
        Z_pred = self._forward(z_t)
        if single:
            return Z_pred[0]
        return Z_pred

    def rollout(self, z0: np.ndarray, horizon: int) -> np.ndarray:
        assert self.fitted, "Model must be fitted first"
        z = z0.copy()
        traj = np.zeros((horizon + 1, self.latent_dim))
        traj[0] = z
        for t in range(horizon):
            z = self.predict_next(z)
            traj[t + 1] = z
        return traj