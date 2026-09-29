import numpy as np


class CandidateModel:

    def __init__(
        self,
        latent_dim: int,
        seed: int = 0,
    ):
        self.latent_dim = int(latent_dim)
        self.seed = int(seed)

        if self.latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive"
            )

        if self.latent_dim != 16:
            raise ValueError(
                "latent_dim does not match "
                "the approved algorithm contract"
            )

        self.num_regimes = 3
        self.ridge = 1e-6
        self.max_kmeans_iter = 50

        self.centroids = None
        self.operators = None
        self.global_operator = None

        self.fitted = False


    # ========================================================
    # Ridge operator
    # ========================================================

    def _fit_linear_operator(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:

        if X.ndim != 2:
            raise ValueError(
                "X must be 2D"
            )

        if Y.ndim != 2:
            raise ValueError(
                "Y must be 2D"
            )

        if X.shape != Y.shape:
            raise ValueError(
                "X and Y must have "
                "the same shape"
            )

        if X.shape[0] < 2:
            return np.eye(
                self.latent_dim
            )

        covariance = (
            X.T @ X
        )

        scale = (
            np.trace(covariance)
            / max(
                self.latent_dim,
                1,
            )
        )

        regularized = (
            covariance
            + self.ridge
            * max(scale, 1.0)
            * np.eye(
                self.latent_dim
            )
        )

        right_hand_side = (
            X.T @ Y
        )

        coefficient = np.linalg.solve(
            regularized,
            right_hand_side,
        )

        operator = coefficient.T

        return operator


    # ========================================================
    # Deterministic initialization
    # ========================================================

    def _initialize_centroids(
        self,
        X: np.ndarray,
        k: int,
    ) -> np.ndarray:

        n_samples = X.shape[0]

        if n_samples < k:
            k = n_samples

        indices = np.linspace(
            0,
            n_samples - 1,
            k,
            dtype=int,
        )

        return X[
            indices
        ].copy()


    # ========================================================
    # K-means
    # ========================================================

    def _kmeans(
        self,
        X: np.ndarray,
        k: int,
    ):
        if X.shape[0] < k:
            k = X.shape[0]

        centroids = (
            self._initialize_centroids(
                X,
                k,
            )
        )

        labels = np.zeros(
            X.shape[0],
            dtype=int,
        )

        for _ in range(
            self.max_kmeans_iter
        ):

            differences = (
                X[:, None, :]
                - centroids[None, :, :]
            )

            distances = np.sum(
                differences
                * differences,
                axis=2,
            )

            new_labels = np.argmin(
                distances,
                axis=1,
            )

            new_centroids = (
                centroids.copy()
            )

            for j in range(k):

                mask = (
                    new_labels == j
                )

                if np.any(mask):

                    new_centroids[j] = (
                        X[mask].mean(
                            axis=0
                        )
                    )

            if np.array_equal(
                labels,
                new_labels,
            ):

                centroids = (
                    new_centroids
                )

                labels = new_labels

                break

            if np.allclose(
                centroids,
                new_centroids,
                rtol=1e-8,
                atol=1e-10,
            ):

                centroids = (
                    new_centroids
                )

                labels = new_labels

                break

            labels = new_labels
            centroids = new_centroids

        return centroids, labels


    # ========================================================
    # Stability monitor
    # ========================================================

    def _stabilize_operator(
        self,
        operator: np.ndarray,
    ) -> np.ndarray:

        eigenvalues = np.linalg.eigvals(
            operator
        )

        spectral_radius = float(
            np.max(
                np.abs(
                    eigenvalues
                )
            )
        )

        if (
            not np.isfinite(
                spectral_radius
            )
        ):

            return np.eye(
                self.latent_dim
            )

        if spectral_radius <= 1.05:

            return operator

        scale = (
            1.05
            / max(
                spectral_radius,
                1.05,
            )
        )

        stabilized = (
            scale * operator
        )

        return stabilized


    # ========================================================
    # Fit
    # ========================================================

    def fit(
        self,
        z_train,
    ):

        Z = np.asarray(
            z_train,
            dtype=np.float64,
        )

        if Z.ndim != 2:
            raise ValueError(
                "z_train must have "
                "shape (T, latent_dim)"
            )

        if Z.shape[1] != self.latent_dim:
            raise ValueError(
                "latent dimension mismatch"
            )

        if Z.shape[0] < 10:
            raise ValueError(
                "Too few training snapshots"
            )

        if not np.all(
            np.isfinite(Z)
        ):
            raise ValueError(
                "z_train contains "
                "NaN or Inf"
            )

        X = Z[
            :-1
        ]

        Y = Z[
            1:
        ]

        # ----------------------------------------------------
        # Global operator
        # ----------------------------------------------------

        self.global_operator = (
            self._fit_linear_operator(
                X,
                Y,
            )
        )

        self.global_operator = (
            self._stabilize_operator(
                self.global_operator
            )
        )

        # ----------------------------------------------------
        # Number of regimes
        # ----------------------------------------------------

        max_regimes = max(
            1,
            X.shape[0] // 20,
        )

        k = min(
            self.num_regimes,
            max_regimes,
        )

        self.centroids, labels = (
            self._kmeans(
                X,
                k,
            )
        )

        # ----------------------------------------------------
        # Regime operators
        # ----------------------------------------------------

        operators = []

        for regime_id in range(k):

            mask = (
                labels == regime_id
            )

            X_regime = X[
                mask
            ]

            Y_regime = Y[
                mask
            ]

            if X_regime.shape[0] < 3:

                operator = (
                    self.global_operator.copy()
                )

            else:

                operator = (
                    self._fit_linear_operator(
                        X_regime,
                        Y_regime,
                    )
                )

                operator = (
                    self._stabilize_operator(
                        operator
                    )
                )

            operators.append(
                operator
            )

        self.operators = np.stack(
            operators,
            axis=0,
        )

        self.fitted = True

        return self


    # ========================================================
    # Predict one step
    # ========================================================

    def predict_next(
        self,
        z_t,
    ):

        if not self.fitted:

            raise RuntimeError(
                "CandidateModel has not "
                "been fitted."
            )

        z = np.asarray(
            z_t,
            dtype=np.float64,
        ).reshape(-1)

        if z.shape[0] != self.latent_dim:

            raise ValueError(
                "z_t has incorrect dimension"
            )

        if not np.all(
            np.isfinite(z)
        ):

            raise ValueError(
                "z_t contains "
                "NaN or Inf"
            )

        differences = (
            self.centroids
            - z[None, :]
        )

        distances = np.sum(
            differences
            * differences,
            axis=1,
        )

        regime_id = int(
            np.argmin(
                distances
            )
        )

        operator = (
            self.operators[
                regime_id
            ]
        )

        z_next = (
            operator @ z
        )

        if not np.all(
            np.isfinite(z_next)
        ):

            raise FloatingPointError(
                "Candidate prediction "
                "contains NaN or Inf"
            )

        return z_next


    # ========================================================
    # Rollout
    # ========================================================

    def rollout(
        self,
        z0,
        horizon,
    ):

        if not self.fitted:

            raise RuntimeError(
                "CandidateModel has not "
                "been fitted."
            )

        horizon = int(
            horizon
        )

        if horizon < 0:
            raise ValueError(
                "horizon must be non-negative"
            )

        z0 = np.asarray(
            z0,
            dtype=np.float64,
        ).reshape(-1)

        if z0.shape[0] != self.latent_dim:

            raise ValueError(
                "z0 has incorrect dimension"
            )

        trajectory = np.zeros(
            (
                horizon + 1,
                self.latent_dim,
            ),
            dtype=np.float64,
        )

        trajectory[0] = z0

        current = z0.copy()

        for step in range(
            horizon
        ):

            current = (
                self.predict_next(
                    current
                )
            )

            trajectory[
                step + 1
            ] = current

        if not np.all(
            np.isfinite(
                trajectory
            )
        ):

            raise FloatingPointError(
                "rollout contains "
                "NaN or Inf"
            )

        return trajectory
