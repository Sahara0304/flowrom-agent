"""CLI entry point for FlowROM autonomous scientific discovery."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from agents.orchestrator import (  # noqa: E402
    ScientificOrchestrator,
)


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "FlowROM autonomous "
            "scientific discovery loop"
        )
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help=(
            "Maximum number of "
            "research cycles."
        ),
    )

    parser.add_argument(
        "--hypotheses",
        type=int,
        default=3,
        help=(
            "Number of competing "
            "hypotheses per cycle."
        ),
    )

    args = parser.parse_args()

    orchestrator = (
        ScientificOrchestrator(
            project_root=ROOT,
            max_cycles=args.cycles,
            hypotheses_per_cycle=
                args.hypotheses,
        )
    )

    result = (
        orchestrator
        .run()
    )

    output = (
        ROOT
        / "research"
        / "autonomous_discovery_result.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "FLOWROM AUTONOMOUS "
        "SCIENTIFIC DISCOVERY"
    )

    print(
        "=" * 78
    )

    for cycle in result.get(
        "cycles",
        [],
    ):

        print(
            f"\n[Cycle {cycle.get('cycle')}]"
        )

        research = cycle.get(
            "research",
            {},
        )

        print(
            "Research source: "
            f"{research.get('generation_source')} | "
            "papers: "
            f"{research.get('literature_count', 0)}"
        )

        for candidate in cycle.get(
            "candidates",
            [],
        ):

            hypothesis = candidate.get(
                "hypothesis",
                {},
            )

            verdict = candidate.get(
                "verdict",
                {},
            )

            print(
                "  "
                f"{hypothesis.get('id')}: "
                f"{hypothesis.get('name')}"
            )

            print(
                "      verdict = "
                f"{verdict.get('status')}"
            )

            one_step = verdict.get(
                "one_step"
            )

            if isinstance(
                one_step,
                dict,
            ):

                print(
                    "      one-step candidate = "
                    f"{one_step.get('candidate', float('nan')):.6f}"
                )

                print(
                    "      one-step global    = "
                    f"{one_step.get('global', float('nan')):.6f}"
                )

            improved = verdict.get(
                "meaningful_improvement_horizons"
            )

            if improved:

                print(
                    "      improved horizons = "
                    f"{improved}"
                )

            print(
                "      archive = "
                f"{candidate.get('archive')}"
            )

    print(
        "\nSaved: "
        f"{output}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )