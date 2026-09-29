from __future__ import annotations

import json
import sys
from pathlib import Path


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


from tools.model_runner import (
    run_development_experiment,
)


def main():

    model_path = (
        Path(
            "algorithms/discovered"
        )
        / "implementation"
        / "model.py"
    )

    latent_path = (
        Path(
            "data/processed"
        )
        / "latent_trajectory.npy"
    )

    output_dir = (
        Path(
            "experiments"
        )
        / "runs"
        / "candidate_001"
    )

    result_path = (
        output_dir
        / "development_result.json"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FLOWROM DEVELOPMENT EXPERIMENT"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # Checks
    # ========================================================

    if not model_path.exists():

        raise SystemExit(
            f"找不到 model.py:\n"
            f"{model_path}"
        )

    if not latent_path.exists():

        raise SystemExit(
            f"找不到 latent trajectory:\n"
            f"{latent_path}"
        )

    # ========================================================
    # Experiment
    # ========================================================

    print()
    print(
        "[Experiment] "
        "Running development-only experiment..."
    )

    print(
        "[Experiment] "
        "Train: first 80% of development"
    )

    print(
        "[Experiment] "
        "Validation: remaining 20% of development"
    )

    print(
        "[Experiment] "
        "Final test: NOT USED"
    )

    result = (
        run_development_experiment(
            model_path=model_path,
            latent_path=latent_path,
        )
    )

    # ========================================================
    # Save
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Print summary
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "RESULT"
    )

    print(
        "=" * 70
    )

    print()

    if result["status"] != "success":

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

        raise SystemExit(1)

    candidate = (
        result[
            "candidate"
        ]
    )

    global_baseline = (
        result[
            "global_linear_baseline"
        ]
    )

    persistence = (
        result[
            "persistence_baseline"
        ]
    )

    print(
        "One-step validation error:"
    )

    print(
        f"  Candidate: "
        f"{candidate['one_step_error']:.6f}"
    )

    print(
        f"  Global Linear: "
        f"{global_baseline['one_step_error']:.6f}"
    )

    print(
        f"  Persistence: "
        f"{persistence['one_step_error']:.6f}"
    )

    print()

    print(
        "Rollout validation error:"
    )

    print()

    candidate_rollout = (
        candidate[
            "rollout_error"
        ]
    )

    global_rollout = (
        global_baseline[
            "rollout_error"
        ]
    )

    persistence_rollout = (
        persistence[
            "rollout_error"
        ]
    )

    horizons = sorted(
        candidate_rollout.keys(),
        key=int,
    )

    print(
        f"{'Horizon':>10}"
        f"{'Candidate':>15}"
        f"{'Global':>15}"
        f"{'Persistence':>15}"
    )

    print(
        "-" * 60
    )

    for horizon in horizons:

        print(
            f"{horizon:>10}"
            f"{candidate_rollout[horizon]:>15.6f}"
            f"{global_rollout[horizon]:>15.6f}"
            f"{persistence_rollout[horizon]:>15.6f}"
        )

    print()

    print(
        "Result saved to:"
    )

    print(
        f"  {result_path}"
    )


if __name__ == "__main__":
    main()