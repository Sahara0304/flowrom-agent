from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ============================================================
# Linear operator
# ============================================================

def fit_linear_operator(
    Z_current: np.ndarray,
    Z_next: np.ndarray,
    ridge: float = 1e-3,
) -> np.ndarray:
    """
    学习：

        z_{t+1} = A z_t

    使用 ridge-stabilized least squares。

    A = Y X^T (X X^T + lambda I)^(-1)
    """

    X = np.asarray(
        Z_current,
        dtype=np.float64,
    )

    Y = np.asarray(
        Z_next,
        dtype=np.float64,
    )
    r = X.shape[0]

    covariance = X @ X.T

    scale = (
        np.trace(covariance)
        / max(r, 1)
    )

    regularized = (
        covariance
        + ridge * scale * np.eye(r)
    )

    A = np.linalg.solve(
        regularized.T,
        (Y @ X.T).T,
    ).T

    return A


# ============================================================
# Relative operator distance
# ============================================================

def operator_distance(
    A: np.ndarray,
    B: np.ndarray,
) -> float:
    """
    Frobenius relative distance.
    """

    denominator = (
        np.linalg.norm(A, ord="fro")
        + np.linalg.norm(B, ord="fro")
        + 1e-12
    )

    return float(
        np.linalg.norm(
            A - B,
            ord="fro",
        )
        / denominator
    )


# ============================================================
# Analyze local regimes
# ============================================================

def analyze_local_regimes(
    Z: np.ndarray,
    output_dir: Path,
    train_ratio: float = 0.8,
    num_windows: int = 5,
    ridge: float = 1e-8,
) -> dict:
    """
    在 development 数据内部：

        前 train_ratio:
            用于 regime discovery

        后 1-train_ratio:
            保留给 validation

    例如：

        development = 400

        training = 320
        validation = 80

    只在 320 个 training latent states
    上识别局部动力学。

    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    r, n = Z.shape

    n_train = int(
        n * train_ratio
    )

    if n_train < num_windows + 2:
        raise ValueError(
            "training 数据不足以划分局部窗口。"
        )

    Z_train = Z[:, :n_train]

    # --------------------------------------------------------
    # Global operator
    # --------------------------------------------------------

    A_global = fit_linear_operator(
        Z_train[:, :-1],
        Z_train[:, 1:],
        ridge=ridge,
    )

    global_spectral_radius = float(
        np.max(
            np.abs(
                np.linalg.eigvals(
                    A_global
                )
            )
        )
    )

    # --------------------------------------------------------
    # Window size
    # --------------------------------------------------------

    usable_pairs = n_train - 1

    window_size = (
        usable_pairs // num_windows
    )

    if window_size < 5:
        raise ValueError(
            "窗口太短，请减少 num_windows。"
        )

    operators = []

    window_results = []

    # --------------------------------------------------------
    # Local operators
    # --------------------------------------------------------

    for k in range(num_windows):

        start = (
            k * window_size
        )

        end = (
            (k + 1) * window_size
        )

        # 最后一个窗口吸收余数
        if k == num_windows - 1:
            end = usable_pairs

        if end - start < 3:
            continue

        current = Z_train[
            :,
            start:end,
        ]

        next_state = Z_train[
            :,
            start + 1:end + 1,
        ]

        A_local = fit_linear_operator(
            current,
            next_state,
            ridge=ridge,
        )

        operators.append(A_local)

        eigenvalues = np.linalg.eigvals(
            A_local
        )

        spectral_radius = float(
            np.max(
                np.abs(
                    eigenvalues
                )
            )
        )

        local_error = (
            np.linalg.norm(
                next_state
                - A_local @ current
            )
            /
            (
                np.linalg.norm(
                    next_state
                )
                + 1e-12
            )
        )

        distance_to_global = (
            operator_distance(
                A_local,
                A_global,
            )
        )

        window_results.append(
            {
                "window": k,
                "start_snapshot": int(
                    start
                ),
                "end_snapshot": int(
                    end
                ),
                "num_pairs": int(
                    end - start
                ),
                "relative_fit_error": float(
                    local_error
                ),
                "spectral_radius": (
                    spectral_radius
                ),
                "distance_to_global": (
                    distance_to_global
                ),
            }
        )

    # --------------------------------------------------------
    # Pairwise operator variation
    # --------------------------------------------------------

    distances = []

    for i in range(
        len(operators)
    ):
        for j in range(
            i + 1,
            len(operators),
        ):
            distances.append(
                operator_distance(
                    operators[i],
                    operators[j],
                )
            )

    if distances:
        mean_pairwise_distance = float(
            np.mean(distances)
        )

        max_pairwise_distance = float(
            np.max(distances)
        )

        median_pairwise_distance = float(
            np.median(distances)
        )

    else:
        mean_pairwise_distance = 0.0
        max_pairwise_distance = 0.0
        median_pairwise_distance = 0.0

    # --------------------------------------------------------
    # 研究信号
    # --------------------------------------------------------

    # 这里故意不把阈值定义成“科学真理”。
    # 只是一个工程上的 flag，供后面的 Research Agent 参考。

    if (
        median_pairwise_distance
        >= 0.10
    ):
        regime_signal = (
            "strong"
        )

    elif (
        median_pairwise_distance
        >= 0.05
    ):
        regime_signal = (
            "moderate"
        )

    else:
        regime_signal = (
            "weak"
        )

    result = {
        "latent_dimension": int(r),
        "num_development_snapshots": int(
            n
        ),
        "num_dynamics_training_snapshots": int(
            n_train
        ),
        "num_validation_snapshots": int(
            n - n_train
        ),
        "num_windows": len(
            operators
        ),
        "window_size": int(
            window_size
        ),
        "global_spectral_radius": (
            global_spectral_radius
        ),
        "mean_pairwise_operator_distance": (
            mean_pairwise_distance
        ),
        "median_pairwise_operator_distance": (
            median_pairwise_distance
        ),
        "max_pairwise_operator_distance": (
            max_pairwise_distance
        ),
        "regime_signal": regime_signal,
        "windows": window_results,
    }

    # --------------------------------------------------------
    # 保存数值结果
    # --------------------------------------------------------

    result_path = (
        output_dir /
        "regime_analysis.json"
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # 保存局部算子
    # --------------------------------------------------------

    if operators:
        operators_array = np.stack(
            operators,
            axis=0,
        )

        np.save(
            output_dir
            / "local_operators.npy",
            operators_array,
        )

    # Global operator 一并保存
    np.save(
        output_dir
        / "global_operator.npy",
        A_global,
    )

    return result