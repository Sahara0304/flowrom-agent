from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

    Z_full = np.load(
        latent_path
    )

    max_rank = Z_full.shape[0]

    candidate_ranks = [
        10,
        20,
        30,
        40,
        max_rank,
    ]

    candidate_ranks = [
        r
        for r in candidate_ranks
        if r <= max_rank
    ]

    print()
    print("=" * 70)
    print("ROBUSTNESS CHECK: LATENT RANK SWEEP")
    print("=" * 70)

    print(
        f"Full latent dimension: {max_rank}"
    )

    print()

    results = []

    for r in candidate_ranks:

        print(
            f"\n>>> Testing latent rank r = {r}"
        )

        Z = Z_full[:r, :]

        result = analyze_local_regimes(
            Z=Z,
            output_dir=processed_dir,
            train_ratio=0.8,
            num_windows=5,
            ridge=1e-6,
        )

        item = {
            "rank": r,
            "median_operator_distance": (
                result[
                    "median_pairwise_operator_distance"
                ]
            ),
            "mean_operator_distance": (
                result[
                    "mean_pairwise_operator_distance"
                ]
            ),
            "max_operator_distance": (
                result[
                    "max_pairwise_operator_distance"
                ]
            ),
            "global_spectral_radius": (
                result[
                    "global_spectral_radius"
                ]
            ),
            "regime_signal": (
                result[
                    "regime_signal"
                ]
            ),
        }

        results.append(item)

        print(
            f"  median drift = "
            f"{item['median_operator_distance']:.6f}"
        )

        print(
            f"  mean drift   = "
            f"{item['mean_operator_distance']:.6f}"
        )

        print(
            f"  max drift    = "
            f"{item['max_operator_distance']:.6f}"
        )

        print(
            f"  spectral rho = "
            f"{item['global_spectral_radius']:.6f}"
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print()

    print(
        f"{'rank':>8}"
        f"{'median':>15}"
        f"{'mean':>15}"
        f"{'max':>15}"
        f"{'rho':>15}"
    )

    print("-" * 70)

    for item in results:

        print(
            f"{item['rank']:>8}"
            f"{item['median_operator_distance']:>15.6f}"
            f"{item['mean_operator_distance']:>15.6f}"
            f"{item['max_operator_distance']:>15.6f}"
            f"{item['global_spectral_radius']:>15.6f}"
        )


if __name__ == "__main__":
    main()