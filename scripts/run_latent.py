from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from tools.latent_analysis import (
    build_latent_trajectory,
    analyze_latent_dynamics,
)


def main():

    parser = argparse.ArgumentParser(
        description="POD -> Latent Dynamics"
    )

    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
    )

    parser.add_argument(
        "--energy",
        type=float,
        default=0.999,
    )

    args = parser.parse_args()

    processed_dir = Path(
        args.processed_dir
    )

    # --------------------------------------------------------
    # 文件
    # --------------------------------------------------------

    matrix_path = (
        processed_dir
        / "snapshot_matrix_development.npy"
    )

    mean_path = (
        processed_dir
        / "snapshot_mean.npy"
    )

    temporal_basis_path = (
        processed_dir
        / "pod_temporal_basis.npy"
    )

    pod_spectrum_path = (
        processed_dir
        / "pod_spectrum.json"
    )

    # --------------------------------------------------------
    # Latent trajectory
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BUILD LATENT TRAJECTORY")
    print("=" * 60)

    metadata = build_latent_trajectory(
        matrix_path=matrix_path,
        mean_path=mean_path,
        temporal_basis_path=temporal_basis_path,
        pod_spectrum_path=pod_spectrum_path,
        output_dir=processed_dir,
        target_energy=args.energy,
    )

    print(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LATENT DYNAMICS")
    print("=" * 60)

    result = analyze_latent_dynamics(
        latent_path=Path(
            metadata["latent_path"]
        ),
        output_dir=processed_dir,
    )

    dynamics = result[
        "linear_dynamics"
    ]

    print()

    print(
        "Linear dynamics:"
    )

    print(
        "  Train error: "
        f"{dynamics['train_relative_error']:.6f}"
    )

    print(
        "  Validation error: "
        f"{dynamics['validation_relative_error']:.6f}"
    )

    print(
        "  Persistence error: "
        f"{dynamics['persistence_validation_error']:.6f}"
    )

    print(
        "  Spectral radius: "
        f"{dynamics['spectral_radius']:.6f}"
    )

    print()

    print(
        "Autocorrelation:"
    )

    for lag, value in enumerate(
        result["autocorrelation"]
    ):
        print(
            f"  lag {lag:2d}: "
            f"{value:.6f}"
        )

    print()

    print(
        "结果已保存到:"
        f" {processed_dir}"
    )


if __name__ == "__main__":
    main()