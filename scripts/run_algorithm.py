from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.algorithm_agent import AlgorithmAgent  # noqa: E402


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    print("=" * 70)
    print("FLOWROM ALGORITHM AGENT")
    print("=" * 70)

    active_path = ROOT / "research" / "active_hypothesis.json"
    if not active_path.exists():
        print("[AlgorithmAgent] Missing research/active_hypothesis.json")
        return 1

    try:
        active = _load_json(active_path) or {}
        print(
            f"[AlgorithmAgent] Active hypothesis: "
            f"{active.get('id')} | {active.get('name')}"
        )
    except Exception as exc:
        print(f"[AlgorithmAgent] Failed to read active hypothesis: {exc}")
        return 1

    research_dir = ROOT / "research"
    output_dir = ROOT / "algorithms" / "discovered"
    research_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = AlgorithmAgent().run(
            research_dir=research_dir,
            output_dir=output_dir,
        )
    except TypeError:
        try:
            result = AlgorithmAgent().run(
                research_dir=str(research_dir),
                output_dir=str(output_dir),
            )
        except Exception as exc:
            print(f"[AlgorithmAgent] ERROR: {exc}")
            return 1
    except Exception as exc:
        print(f"[AlgorithmAgent] ERROR: {exc}")
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    if not isinstance(result, dict):
        return 1

    status = str(result.get("status", "")).lower()

    # IMPORTANT:
    # not_implementable is a valid scientific/system state, not an execution failure.
    if status in {"success", "not_implementable"}:
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
