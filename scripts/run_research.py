from __future__ import annotations

import json
import sys
from pathlib import Path


# ============================================================
# Project Root
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


from agents.research_agent import (
    ResearchAgent,
)


def main():

    processed_dir = Path(
        "data/processed"
    )

    output_dir = Path(
        "research/candidates"
    )

    # ========================================================
    # 检查实验结果是否存在
    # ========================================================

    required_files = [
        (
            processed_dir
            / "pod_spectrum.json"
        ),
        (
            processed_dir
            / "latent_dynamics.json"
        ),
        (
            processed_dir
            / "regime_analysis.json"
        ),
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:

        print()
        print(
            "[ERROR] Missing required "
            "experiment files:"
        )

        for path in missing:
            print(
                f"  - {path}"
            )

        print()
        print(
            "请先完成前面的数据分析流程。"
        )

        raise SystemExit(1)

    # ========================================================
    # Research Agent
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FLOWROM RESEARCH AGENT"
    )

    print(
        "=" * 70
    )

    agent = ResearchAgent(
        model="MiniMax-M3"
    )

    result = agent.run(
        processed_dir=processed_dir,
        output_dir=output_dir,
    )

    # ========================================================
    # Print Summary
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "RESEARCH RESULT"
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
        "=" * 70
    )

    print(
        "FILES"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "Literature:"
    )

    print(
        "  research/candidates/"
        "literature_results.json"
    )

    print()
    print(
        "Research state:"
    )

    print(
        "  research/candidates/"
        "research_state.json"
    )

    print()
    print(
        "Hypotheses:"
    )

    print(
        "  research/candidates/"
        "research_hypotheses.json"
    )

    print()

    raw_response = (
        output_dir
        / "research_raw_response.txt"
    )

    if raw_response.exists():

        print(
            "Raw LLM response:"
        )

        print(
            "  research/candidates/"
            "research_raw_response.txt"
        )


if __name__ == "__main__":
    main()