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

from runtime.algorithm_contract import validate_algorithm_contract
from tools.operator_analysis import DiagnosticDataError, load_latent_trajectory, operator_decomposition


def _save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    print("=" * 70)
    print("FLOWROM DIAGNOSTIC ANALYSIS")
    print("=" * 70)

    spec_path = ROOT / "algorithms" / "discovered" / "algorithm_spec.json"
    output_path = ROOT / "experiments" / "runs" / "candidate_001" / "diagnostic_result.json"

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        validate_algorithm_contract(spec)
    except Exception as exc:
        result = {"status": "contract_load_failed", "error": str(exc)}
        _save(result, output_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    candidate_id = spec.get("candidate_id")
    experiment = spec.get("experiment", {}) or {}
    analysis_type = experiment.get("analysis_type")

    if spec.get("execution_mode") != "analysis_runner":
        result = {
            "status": "not_implementable",
            "candidate_id": candidate_id,
            "error": "Algorithm contract is not configured for analysis_runner.",
        }
        _save(result, output_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if analysis_type != "operator_decomposition":
        result = {
            "status": "not_implementable",
            "candidate_id": candidate_id,
            "error": f"Unsupported diagnostic analysis: {analysis_type}",
        }
        _save(result, output_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    try:
        latent, latent_path = load_latent_trajectory()
        result = operator_decomposition(latent)
        result["candidate_id"] = candidate_id
        result["candidate_name"] = spec.get("name")
        result["source"]["latent_path"] = str(latent_path)
        _save(result, output_path)
    except DiagnosticDataError as exc:
        result = {
            "status": "not_implementable",
            "candidate_id": candidate_id,
            "error": str(exc),
        }
        _save(result, output_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        result = {
            "status": "diagnostic_execution_failed",
            "candidate_id": candidate_id,
            "error": str(exc),
        }
        _save(result, output_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
