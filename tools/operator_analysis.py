from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LATENT_DIM = 52
TRAIN_END = 320
DEV_END = 400


class DiagnosticDataError(RuntimeError):
    pass


def _candidate_paths() -> list[Path]:
    processed = ROOT / "data" / "processed"
    return [
        processed / "latent_trajectory.npy",
        processed / "latent.npy",
        processed / "pod_latent.npy",
        processed / "development_latent.npy",
        processed / "development_latent_trajectory.npy",
    ]


def load_latent_trajectory() -> tuple[np.ndarray, Path]:
    """Load an n x latent_dim trajectory without touching final-test data."""
    for path in _candidate_paths():
        if not path.exists():
            continue
        try:
            raw = np.load(path, mmap_mode="r")
        except Exception:
            continue
        if raw.ndim == 2 and raw.shape[1] == LATENT_DIM and raw.shape[0] >= DEV_END:
            return np.array(raw[:DEV_END], dtype=float, copy=True), path

    processed = ROOT / "data" / "processed"
    matches: list[tuple[Path, np.ndarray]] = []
    if processed.exists():
        for path in sorted(processed.rglob("*.npy")):
            try:
                arr = np.asarray(np.load(path, mmap_mode="r"))
            except Exception:
                continue
            if arr.ndim == 2 and arr.shape[1] == LATENT_DIM and arr.shape[0] >= DEV_END:
                matches.append((path, arr))

    if len(matches) == 1:
        return np.asarray(np.load(matches[0][0]), dtype=float), matches[0][0]

    if matches:
        preferred = [
            item for item in matches
            if any(token in item[0].stem.lower() for token in ("latent", "pod"))
        ]
        if len(preferred) == 1:
            raw = np.load(preferred[0][0], mmap_mode="r")
            return np.array(raw[:DEV_END], dtype=float, copy=True), preferred[0][0]

    candidates = ", ".join(str(p) for p in _candidate_paths())
    raise DiagnosticDataError(
        "Cannot uniquely locate a 52D latent trajectory. "
        f"Expected one of: {candidates}"
    )


def fit_global_operator(z_train: np.ndarray) -> np.ndarray:
    if z_train.ndim != 2 or z_train.shape[1] != LATENT_DIM:
        raise DiagnosticDataError("Expected latent matrix shape (n_samples, 52).")
    x = z_train[:-1]
    y = z_train[1:]
    gram = x.T @ x
    scale = max(float(np.trace(gram) / LATENT_DIM), 1.0)
    ridge = 1e-8 * scale
    coef = np.linalg.solve(
        gram + ridge * np.eye(LATENT_DIM),
        x.T @ y,
    )
    return coef.T


def relative_frobenius(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, "fro"))


def _safe_fraction(num: float, den: float) -> float:
    if den <= 1e-15:
        return 0.0
    return float(num / den)


def operator_decomposition(z: np.ndarray) -> dict[str, Any]:
    """Run the first diagnostic experiment on development data only.

    A is fitted on snapshots 1..320. Validation diagnostics use 321..400 only.
    The diagnostic decomposes A = S + K where S=(A+A^T)/2 and K=(A-A^T)/2.
    POD coordinates are partitioned into three contiguous energy bands.
    """
    if z.shape[0] < DEV_END:
        raise DiagnosticDataError(
            f"Need at least {DEV_END} development snapshots; got {z.shape[0]}"
        )

    z_train = z[:TRAIN_END]
    z_val = z[TRAIN_END:DEV_END]
    if z_train.shape[0] < 3 or z_val.shape[0] < 2:
        raise DiagnosticDataError("Insufficient development train/validation samples.")

    A = fit_global_operator(z_train)
    S = 0.5 * (A + A.T)
    K = 0.5 * (A - A.T)

    a_norm = relative_frobenius(A)
    s_norm = relative_frobenius(S)
    k_norm = relative_frobenius(K)

    # Because POD coordinates are ordered by energy, contiguous bands are a
    # transparent first diagnostic rather than a tuned segmentation.
    edges = [0, LATENT_DIM // 3, 2 * LATENT_DIM // 3, LATENT_DIM]
    bands = []
    for i in range(3):
        lo, hi = edges[i], edges[i + 1]
        idx = np.arange(lo, hi)
        P = np.zeros((LATENT_DIM, LATENT_DIM))
        P[idx, idx] = 1.0
        Ab = P @ A @ P
        Sb = 0.5 * (Ab + Ab.T)
        Kb = 0.5 * (Ab - Ab.T)
        abn = relative_frobenius(Ab)
        sbn = relative_frobenius(Sb)
        kbn = relative_frobenius(Kb)
        bands.append(
            {
                "band": i + 1,
                "index_start": int(lo),
                "index_end_exclusive": int(hi),
                "operator_norm": abn,
                "symmetric_norm": sbn,
                "skew_norm": kbn,
                "symmetric_fraction": _safe_fraction(sbn, abn),
                "skew_fraction": _safe_fraction(kbn, abn),
            }
        )

    # Validation one-step error is included as context, not as a model comparison.
    x_val = z_val[:-1]
    y_val = z_val[1:]
    y_hat = (A @ x_val.T).T
    denom = max(float(np.linalg.norm(y_val)), 1e-12)
    val_one_step = float(np.linalg.norm(y_val - y_hat) / denom)

    band_skew = [b["skew_fraction"] for b in bands]
    band_sym = [b["symmetric_fraction"] for b in bands]

    return {
        "status": "diagnostic_complete",
        "candidate_type": "diagnostic_analysis",
        "analysis_type": "operator_decomposition",
        "data_policy": {
            "used_splits": ["development_train", "development_validation"],
            "final_test_access": False,
        },
        "source": {
            "latent_shape": list(z.shape),
            "latent_dimension": LATENT_DIM,
            "train_snapshots": [1, TRAIN_END],
            "validation_snapshots": [TRAIN_END + 1, DEV_END],
        },
        "global": {
            "operator_frobenius_norm": a_norm,
            "symmetric_norm": s_norm,
            "skew_norm": k_norm,
            "symmetric_fraction": _safe_fraction(s_norm, a_norm),
            "skew_fraction": _safe_fraction(k_norm, a_norm),
            "symmetric_plus_skew_reconstruction_error": float(
                np.linalg.norm(A - (S + K), "fro") / max(a_norm, 1e-12)
            ),
        },
        "bands": bands,
        "validation_context": {
            "global_linear_one_step_relative_error": val_one_step,
        },
        "diagnostic_summary": {
            "nontrivial_skew_component": bool(_safe_fraction(k_norm, a_norm) > 1e-3),
            "nontrivial_symmetric_component": bool(_safe_fraction(s_norm, a_norm) > 1e-3),
            "bandwise_skew_range": float(max(band_skew) - min(band_skew)),
            "bandwise_symmetric_range": float(max(band_sym) - min(band_sym)),
        },
    }


def save_result(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
