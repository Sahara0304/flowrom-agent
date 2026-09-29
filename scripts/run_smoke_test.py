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
    run_smoke_test,
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
            "algorithms/discovered"
        )
        / "implementation"
    )

    result_path = (
        output_dir
        / "smoke_test.json"
    )

    print()
    print(
        "=" * 70
    )
    print(
        "FLOWROM SMOKE TEST"
    )
    print(
        "=" * 70
    )

    if not model_path.exists():

        raise SystemExit(
            f"找不到 model.py:\n"
            f"{model_path}\n\n"
            "请先运行：\n"
            "python scripts/run_code.py"
        )

    if not latent_path.exists():

        raise SystemExit(
            f"找不到 latent trajectory:\n"
            f"{latent_path}"
        )

    result = run_smoke_test(
        model_path=model_path,
        latent_path=latent_path,
    )

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

    print()

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()

    if result["status"] == "passed":

        print(
            "[SUCCESS] "
            "Smoke test passed."
        )

    else:

        print(
            "[FAILED] "
            "Smoke test failed."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()