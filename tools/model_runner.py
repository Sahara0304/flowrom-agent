from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================
# Load candidate model
# ============================================================

def load_candidate_model(
    model_path: Path,
):
    """
    动态加载：

        model.py

    并获取：

        CandidateModel
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{model_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "candidate_model",
        model_path,
    )

    if spec is None:
        raise ImportError(
            "无法创建 module spec。"
        )

    if spec.loader is None:
        raise ImportError(
            "模型文件没有可用 loader。"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    if not hasattr(
        module,
        "CandidateModel",
    ):
        raise AttributeError(
            "model.py 必须定义 CandidateModel。"
        )

    CandidateModel = (
        module.CandidateModel
    )

    return CandidateModel


# ============================================================
# Validation helpers
# ============================================================

def assert_finite(
    x: np.ndarray,
    name: str,
) -> None:

    if not np.all(
        np.isfinite(x)
    ):
        raise ValueError(
            f"{name} 包含 NaN 或 Inf。"
        )


def relative_l2_error(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:

    numerator = np.linalg.norm(
        prediction - target
    )

    denominator = (
        np.linalg.norm(target)
        + 1e-12
    )

    return float(
        numerator / denominator
    )


# ============================================================
# Smoke Test
# ============================================================

def run_smoke_test(
    model_path: Path,
    latent_path: Path,
) -> dict[str, Any]:

    try:

        # ----------------------------------------------------
        # Load latent data
        # ----------------------------------------------------

        Z = np.load(
            latent_path
        )

        if Z.ndim != 2:
            raise ValueError(
                f"latent trajectory 必须为二维，"
                f"当前 shape={Z.shape}"
            )

        # 原始：
        #     (r, T)
        #
        # 转为：
        #     (T, r)

        if Z.shape[0] < Z.shape[1]:

            Z = Z.T

        T, latent_dim = Z.shape

        if T < 20:
            raise ValueError(
                "snapshot 数量太少，"
                "无法进行 smoke test。"
            )

        assert_finite(
            Z,
            "latent trajectory",
        )

        # ----------------------------------------------------
        # Development split
        # ----------------------------------------------------

        n_train = int(
            T * 0.8
        )

        if n_train < 10:
            raise ValueError(
                "development_train 太短。"
            )

        Z_train = Z[
            :n_train
        ]

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        CandidateModel = (
            load_candidate_model(
                model_path
            )
        )

        model = CandidateModel(
            latent_dim=latent_dim,
            seed=0,
        )

        # ----------------------------------------------------
        # Fit
        # ----------------------------------------------------

        model.fit(
            Z_train
        )

        # ----------------------------------------------------
        # One-step prediction
        # ----------------------------------------------------

        z0 = Z_train[
            -1
        ]

        prediction = model.predict_next(
            z0
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float64,
        )

        if prediction.shape != (
            latent_dim,
        ):
            raise ValueError(
                "predict_next 输出 shape 错误："
                f"{prediction.shape}, "
                f"期望 {(latent_dim,)}"
            )

        assert_finite(
            prediction,
            "predict_next",
        )

        # ----------------------------------------------------
        # Rollout
        # ----------------------------------------------------

        horizon = min(
            10,
            T - n_train,
        )

        rollout = model.rollout(
            z0,
            horizon,
        )

        rollout = np.asarray(
            rollout,
            dtype=np.float64,
        )

        expected_shape = (
            horizon + 1,
            latent_dim,
        )

        if rollout.shape != expected_shape:
            raise ValueError(
                "rollout 输出 shape 错误："
                f"{rollout.shape}, "
                f"期望 {expected_shape}"
            )

        assert_finite(
            rollout,
            "rollout",
        )

        # 第一状态应该等于 z0
        initial_error = np.linalg.norm(
            rollout[0] - z0
        )

        if initial_error > 1e-8:
            raise ValueError(
                "rollout[0] 必须等于 z0。"
            )

        return {
            "status": "passed",
            "latent_shape": list(
                Z.shape
            ),
            "latent_dimension": (
                int(latent_dim)
            ),
            "train_snapshots": int(
                n_train
            ),
            "one_step_shape": list(
                prediction.shape
            ),
            "rollout_shape": list(
                rollout.shape
            ),
            "max_abs_rollout": float(
                np.max(
                    np.abs(rollout)
                )
            ),
        }

    except Exception as exc:

        return {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# Global Linear Baseline
# ============================================================

def fit_global_linear_baseline(
    Z_train: np.ndarray,
    ridge: float = 1e-6,
) -> np.ndarray:

    X = Z_train[
        :-1
    ].T

    Y = Z_train[
        1:
    ].T

    r = X.shape[0]

    covariance = (
        X @ X.T
    )

    scale = (
        np.trace(covariance)
        / max(r, 1)
    )

    regularized = (
        covariance
        + ridge
        * scale
        * np.eye(r)
    )

    A = np.linalg.solve(
        regularized.T,
        (Y @ X.T).T,
    ).T

    return A


# ============================================================
# Candidate one-step validation
# ============================================================

def evaluate_one_step(
    model,
    Z_all: np.ndarray,
    n_train: int,
) -> tuple[float, np.ndarray]:

    current = Z_all[
        n_train - 1:-1
    ]

    target = Z_all[
        n_train:
    ]

    predictions = []

    for z in current:

        pred = model.predict_next(
            z
        )

        pred = np.asarray(
            pred,
            dtype=np.float64,
        )

        predictions.append(
            pred
        )

    predictions = np.stack(
        predictions,
        axis=0,
    )

    assert_finite(
        predictions,
        "candidate validation predictions",
    )

    error = relative_l2_error(
        predictions,
        target,
    )

    return error, predictions


# ============================================================
# Persistence baseline
# ============================================================

def persistence_predictions(
    current: np.ndarray,
) -> np.ndarray:

    return np.asarray(
        current,
        dtype=np.float64,
    ).copy()


# ============================================================
# Rollout evaluation
# ============================================================

def evaluate_rollout(
    model,
    initial_state: np.ndarray,
    target_states: np.ndarray,
    horizons: list[int],
) -> dict[str, float]:

    results = {}

    max_horizon = len(
        target_states
    )

    for horizon in horizons:

        if horizon > max_horizon:
            continue

        prediction = model.rollout(
            initial_state,
            horizon,
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float64,
        )

        expected_shape = (
            horizon + 1,
            initial_state.shape[0],
        )

        if prediction.shape != expected_shape:

            raise ValueError(
                "rollout shape mismatch："
                f"{prediction.shape} "
                f"!= "
                f"{expected_shape}"
            )

        assert_finite(
            prediction,
            f"rollout horizon={horizon}",
        )

        target = target_states[
            :horizon
        ]

        error = relative_l2_error(
            prediction[1:],
            target,
        )

        results[
            str(horizon)
        ] = error

    return results


# ============================================================
# Experiment
# ============================================================

def run_development_experiment(
    model_path: Path,
    latent_path: Path,
) -> dict[str, Any]:

    try:

        # ----------------------------------------------------
        # Load latent
        # ----------------------------------------------------

        Z = np.load(
            latent_path
        )

        if Z.ndim != 2:
            raise ValueError(
                "latent trajectory 必须二维。"
            )

        if Z.shape[0] < Z.shape[1]:
            Z = Z.T

        T, latent_dim = Z.shape

        assert_finite(
            Z,
            "latent trajectory",
        )

        # ----------------------------------------------------
        # Development split
        # ----------------------------------------------------

        n_train = int(
            T * 0.8
        )

        if (
            n_train <= 1
            or T - n_train <= 1
        ):
            raise ValueError(
                "train / validation 划分无效。"
            )

        Z_train = Z[
            :n_train
        ]

        Z_validation = Z[
            n_train:
        ]

        # ----------------------------------------------------
        # Candidate
        # ----------------------------------------------------

        CandidateModel = (
            load_candidate_model(
                model_path
            )
        )

        model = CandidateModel(
            latent_dim=latent_dim,
            seed=0,
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            Z_train
        )

        # ----------------------------------------------------
        # Candidate one-step
        # ----------------------------------------------------

        candidate_one_step_error, candidate_predictions = (
            evaluate_one_step(
                model,
                Z,
                n_train,
            )
        )

        # ----------------------------------------------------
        # Persistence one-step
        # ----------------------------------------------------

        persistence_current = Z[
            n_train - 1:-1
        ]

        persistence_target = Z[
            n_train:
        ]

        persistence_pred = (
            persistence_predictions(
                persistence_current
            )
        )

        persistence_error = (
            relative_l2_error(
                persistence_pred,
                persistence_target,
            )
        )

        # ----------------------------------------------------
        # Global linear baseline
        # ----------------------------------------------------

        A = fit_global_linear_baseline(
            Z_train
        )

        global_current = Z[
            n_train - 1:-1
        ]

        global_target = Z[
            n_train:
        ]

        global_prediction = (
            global_current @ A.T
        )

        global_one_step_error = (
            relative_l2_error(
                global_prediction,
                global_target,
            )
        )

        # ----------------------------------------------------
        # Rollout
        # ----------------------------------------------------

        initial_state = Z[
            n_train - 1
        ]

        horizons = [
            1,
            5,
            10,
            20,
            40,
            80,
        ]

        candidate_rollout = (
            evaluate_rollout(
                model,
                initial_state,
                Z_validation,
                horizons,
            )
        )

        # ----------------------------------------------------
        # Global baseline rollout
        # ----------------------------------------------------

        global_rollout_errors = {}

        for horizon in horizons:

            if horizon > len(
                Z_validation
            ):
                continue

            trajectory = [
                initial_state.copy()
            ]

            current = (
                initial_state.copy()
            )

            for _ in range(
                horizon
            ):

                current = A @ current

                trajectory.append(
                    current.copy()
                )

            trajectory = np.stack(
                trajectory,
                axis=0,
            )

            target = Z_validation[
                :horizon
            ]

            global_rollout_errors[
                str(horizon)
            ] = relative_l2_error(
                trajectory[1:],
                target,
            )

        # ----------------------------------------------------
        # Persistence rollout
        # ----------------------------------------------------

        persistence_rollout_errors = {}

        for horizon in horizons:

            if horizon > len(
                Z_validation
            ):
                continue

            trajectory = np.repeat(
                initial_state[
                    None,
                    :,
                ],
                horizon + 1,
                axis=0,
            )

            target = Z_validation[
                :horizon
            ]

            persistence_rollout_errors[
                str(horizon)
            ] = relative_l2_error(
                trajectory[1:],
                target,
            )

        # ----------------------------------------------------
        # Baseline comparison
        # ----------------------------------------------------

        comparison = {
            "candidate_one_step": (
                candidate_one_step_error
            ),
            "global_linear_one_step": (
                global_one_step_error
            ),
            "persistence_one_step": (
                persistence_error
            ),
            "candidate_vs_global_one_step_ratio": (
                candidate_one_step_error
                /
                (
                    global_one_step_error
                    + 1e-12
                )
            ),
            "candidate_vs_persistence_one_step_ratio": (
                candidate_one_step_error
                /
                (
                    persistence_error
                    + 1e-12
                )
            ),
        }

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        return {

            "status": "success",

            "data_policy": {
                "development_only": True,
                "final_test_used": False,
                "train_range": [
                    0,
                    n_train - 1,
                ],
                "validation_range": [
                    n_train,
                    T - 1,
                ],
            },

            "data": {
                "num_snapshots": int(T),
                "latent_dimension": int(
                    latent_dim
                ),
                "train_snapshots": int(
                    n_train
                ),
                "validation_snapshots": int(
                    T - n_train
                ),
            },

            "candidate": {
                "one_step_error": (
                    candidate_one_step_error
                ),
                "rollout_error": (
                    candidate_rollout
                ),
            },

            "global_linear_baseline": {
                "one_step_error": (
                    global_one_step_error
                ),
                "rollout_error": (
                    global_rollout_errors
                ),
            },

            "persistence_baseline": {
                "one_step_error": (
                    persistence_error
                ),
                "rollout_error": (
                    persistence_rollout_errors
                ),
            },

            "comparison": comparison,
        }

    except Exception as exc:

        return {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# Save JSON
# ============================================================

def save_result(
    result: dict[str, Any],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )