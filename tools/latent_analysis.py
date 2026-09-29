from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ============================================================
# POD -> Latent Trajectory
# ============================================================

def build_latent_trajectory(
    matrix_path: Path,
    mean_path: Path,
    temporal_basis_path: Path,
    pod_spectrum_path: Path,
    output_dir: Path,
    target_energy: float = 0.999,
) -> dict:
    """
    根据 POD 结果构造 latent trajectory。

    数学上：

        X_c = U Sigma V^T

    因此：

        Z = Sigma_r V_r^T

    其中 Z[:, t] = z_t
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    X = np.load(
        matrix_path,
        mmap_mode="r",
    )

    mean = np.load(mean_path)

    V = np.load(
        temporal_basis_path
    )

    with open(
        pod_spectrum_path,
        "r",
        encoding="utf-8",
    ) as f:
        pod_info = json.load(f)

    singular_values = np.asarray(
        pod_info["singular_values"],
        dtype=np.float64,
    )

    cumulative_energy = np.asarray(
        pod_info["cumulative_energy"],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # 自动确定 latent dimension
    # --------------------------------------------------------

    rank = int(
        np.searchsorted(
            cumulative_energy,
            target_energy,
        )
        + 1
    )

    rank = min(
        rank,
        len(singular_values),
    )

    print(
        f"Latent dimension r = {rank}"
    )

    # --------------------------------------------------------
    # Z = Sigma_r V_r^T
    #
    # shape:
    #     Sigma_r : (r,)
    #     V_r.T   : (r, N)
    #
    # 得到：
    #     Z : (r, N)
    # --------------------------------------------------------

    sigma_r = singular_values[:rank]

    V_r = V[:, :rank]

    Z = (
        sigma_r[:, None]
        * V_r.T
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    latent_path = (
        output_dir /
        "latent_trajectory.npy"
    )

    np.save(
        latent_path,
        Z,
    )

    metadata = {
        "latent_dimension": rank,
        "num_snapshots": int(Z.shape[1]),
        "target_energy": target_energy,
        "trajectory_shape": [
            int(Z.shape[0]),
            int(Z.shape[1]),
        ],
        "definition": "Z = Sigma_r @ V_r.T",
        "latent_path": str(
            latent_path
        ),
    }

    metadata_path = (
        output_dir /
        "latent_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return metadata


# ============================================================
# 时间自相关
# ============================================================

def compute_autocorrelation(
    z: np.ndarray,
    max_lag: int = 10,
) -> list[float]:
    """
    计算整体 latent state 的归一化自相关。
    """

    # z:
    # (r, N)

    z_centered = (
        z
        - z.mean(axis=1, keepdims=True)
    )

    values = []

    denominator = np.sum(
        z_centered ** 2
    )

    if denominator <= 0:
        return [
            1.0
        ] + [
            0.0
            for _ in range(max_lag)
        ]

    for lag in range(
        max_lag + 1
    ):

        if lag == 0:
            numerator = denominator

        else:
            numerator = np.sum(
                z_centered[:, :-lag]
                *
                z_centered[:, lag:]
            )

        values.append(
            float(
                numerator
                / denominator
            )
        )

    return values


# ============================================================
# Linear latent dynamics
# ============================================================

def fit_linear_dynamics(
    Z: np.ndarray,
    train_ratio: float = 0.8,
) -> dict:
    """
    拟合：

        z_{t+1} = A z_t

    并在 development 数据内部做
    train / validation。

    注意：
        最后的 50% test 数据根本没有进入这里。
    """

    r, n = Z.shape

    n_train = int(
        n * train_ratio
    )

    if n_train < 2:
        raise ValueError(
            "训练 snapshot 太少。"
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    Z_current = Z[
        :, :n_train - 1
    ]

    Z_next = Z[
        :, 1:n_train
    ]

    # A = Y X^+
    A = (
        Z_next
        @ np.linalg.pinv(
            Z_current
        )
    )

    # --------------------------------------------------------
    # Training prediction
    # --------------------------------------------------------

    Z_train_pred = (
        A @ Z_current
    )

    train_error = (
        np.linalg.norm(
            Z_next
            - Z_train_pred
        )
        /
        np.linalg.norm(
            Z_next
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    Z_val_current = Z[
        :,
        n_train - 1:-1,
    ]

    Z_val_next = Z[
        :,
        n_train:,
    ]

    Z_val_pred = (
        A @ Z_val_current
    )

    val_error = (
        np.linalg.norm(
            Z_val_next
            - Z_val_pred
        )
        /
        np.linalg.norm(
            Z_val_next
        )
    )

    # --------------------------------------------------------
    # Persistence baseline
    #
    # z_{t+1} = z_t
    # --------------------------------------------------------

    persistence_error = (
        np.linalg.norm(
            Z_val_next
            - Z_val_current
        )
        /
        np.linalg.norm(
            Z_val_next
        )
    )

    # --------------------------------------------------------
    # A 的谱半径
    # --------------------------------------------------------

    eigenvalues = np.linalg.eigvals(
        A
    )

    spectral_radius = float(
        np.max(
            np.abs(
                eigenvalues
            )
        )
    )

    return {
        "latent_dimension": int(r),
        "num_snapshots": int(n),
        "num_train_snapshots": int(
            n_train
        ),
        "num_validation_snapshots": int(
            n - n_train
        ),
        "train_relative_error": float(
            train_error
        ),
        "validation_relative_error": float(
            val_error
        ),
        "persistence_validation_error": float(
            persistence_error
        ),
        "spectral_radius": spectral_radius,
        "linear_operator": A.tolist(),
    }


# ============================================================
# 整体分析
# ============================================================

def analyze_latent_dynamics(
    latent_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    max_lag: int = 20,
) -> dict:

    Z = np.load(
        latent_path
    )

    autocorrelation = (
        compute_autocorrelation(
            Z,
            max_lag=max_lag,
        )
    )

    dynamics = fit_linear_dynamics(
        Z,
        train_ratio=train_ratio,
    )

    result = {
        "latent_shape": [
            int(Z.shape[0]),
            int(Z.shape[1]),
        ],
        "autocorrelation": autocorrelation,
        "linear_dynamics": dynamics,
    }

    result_path = (
        output_dir /
        "latent_dynamics.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return result