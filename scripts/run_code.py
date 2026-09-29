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


from agents.code_agent import (
    CodeAgent,
)


def main():

    # ========================================================
    # Approved contract
    # ========================================================

    spec_path = (
        Path(
            "algorithms/discovered"
        )
        / "algorithm_spec.json"
    )

    output_dir = (
        Path(
            "algorithms/discovered"
        )
        / "implementation"
    )

    print()
    print(
        "=" * 70
    )
    print(
        "FLOWROM CODE AGENT"
    )
    print(
        "=" * 70
    )

    if not spec_path.exists():

        print()

        print(
            "[ERROR] "
            "Approved algorithm contract not found:"
        )

        print(
            f"  {spec_path}"
        )

        print()

        print(
            "请先运行："
        )

        print(
            "  python scripts/run_algorithm.py"
        )

        raise SystemExit(1)

    agent = CodeAgent(
        model="MiniMax-M3"
    )

    result = agent.run(
        spec_path=spec_path,
        output_dir=output_dir,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "CODE RESULT"
    )
    print(
        "=" * 70
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

    print(
        "Output:"
    )

    print(
        f"  {output_dir}"
    )


if __name__ == "__main__":
    main()