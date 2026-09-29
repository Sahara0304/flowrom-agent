from __future__ import annotations

import json
import sys
from pathlib import Path


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import numpy as np

from tools.regime_analysis import (
    analyze_local_regimes,
)


def main():

    processed_dir = Path(
        "data/processed"
    )

    latent_path = (
        processed_dir
        / "latent_trajectory.npy"
    )

    if not latent_path.exists():

        raise FileNotFoundError(
            f"找不到: {latent_path}\n"
            "请先运行:\n"
            "python scripts/run_latent.py"
        )

    Z = np.load(
        latent_path
    )

    print()
    print("=" * 60)
    print("LOCAL DYNAMICS / REGIME ANALYSIS")
    print("=" * 60)

    print(
        f"Latent shape: {Z.shape}"
    )

    result = analyze_local_regimes(
        Z=Z,
        output_dir=processed_dir,
        train_ratio=0.8,
        num_windows=5,
    )

    print()

    print(
        "Global spectral radius: "
        f"{result['global_spectral_radius']:.6f}"
    )

    print()

    print(
        "Operator variation:"
    )

    print(
        "  Mean:   "
        f"{result['mean_pairwise_operator_distance']:.6f}"
    )

    print(
        "  Median: "
        f"{result['median_pairwise_operator_distance']:.6f}"
    )

    print(
        "  Max:    "
        f"{result['max_pairwise_operator_distance']:.6f}"
    )

    print()

    print(
        "Regime signal: "
        f"{result['regime_signal']}"
    )

    print()

    for item in result["windows"]:

        print(
            f"Window {item['window']}: "
            f"fit={item['relative_fit_error']:.6f}, "
            f"distance={item['distance_to_global']:.6f}, "
            f"rho={item['spectral_radius']:.6f}"
        )

    print()

    print(
        "结果已保存到:"
        f" {processed_dir / 'regime_analysis.json'}"
    )


if __name__ == "__main__":
    main()