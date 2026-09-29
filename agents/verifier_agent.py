"""Scientific result verifier for FlowROM-Agent."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from research.research_state import record_hypothesis_outcome, save_json


class VerifierAgent:
    """Convert experiment outcomes into scientific/system-level verdicts.

    Important distinction:
    - not_implementable = capability mismatch, not scientific evidence
    - execution_failed = infrastructure/code failure
    - falsified/supported/inconclusive = scientific evidence after execution
    """

    def __init__(self, project_root: Optional[Path] = None):
        self.root = Path(project_root or Path(__file__).resolve().parents[1])

    @staticmethod
    def _find_one_step(obj: Any) -> Optional[Tuple[float, float]]:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if "one_step" in str(key).lower() and isinstance(value, dict):
                    candidate = value.get("candidate")
                    global_value = value.get("global_linear", value.get("global"))
                    if isinstance(candidate, (int, float)) and isinstance(global_value, (int, float)):
                        return float(candidate), float(global_value)
                found = VerifierAgent._find_one_step(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = VerifierAgent._find_one_step(value)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _find_rollout(obj: Any) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, list):
                    key_lower = str(key).lower()
                    if "rollout" in key_lower or "horizon" in key_lower:
                        for row in value:
                            if not isinstance(row, dict):
                                continue
                            horizon = row.get("horizon")
                            candidate = row.get("candidate")
                            global_value = row.get("global_linear", row.get("global"))
                            if all(isinstance(v, (int, float)) for v in (horizon, candidate, global_value)):
                                rows.append({
                                    "horizon": float(horizon),
                                    "candidate": float(candidate),
                                    "global": float(global_value),
                                })
                rows.extend(VerifierAgent._find_rollout(value))
        elif isinstance(obj, list):
            for value in obj:
                rows.extend(VerifierAgent._find_rollout(value))
        unique = {row["horizon"]: row for row in rows}
        return [unique[h] for h in sorted(unique)]

    @staticmethod
    def _load_execution_metadata(root: Path) -> Dict[str, Any]:
        return {
            "algorithm_spec": _load(root / "algorithms/discovered/algorithm_spec.json"),
            "algorithm_result": _load(root / "algorithms/discovered/algorithm_result.json"),
            "code_result": _load(root / "algorithms/discovered/implementation/code_result.json"),
        }

    def verify(
        self,
        hypothesis: Dict[str, Any],
        experiment_result: Optional[Dict[str, Any]],
        execution_ok: bool = True,
        execution_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        # ---------------------------------------------------------
        # Capability mismatch: not a scientific failure.
        # ---------------------------------------------------------
        if isinstance(experiment_result, dict) and experiment_result.get("status") == "not_implementable":
            implementability = experiment_result.get("implementability", {})
            missing = experiment_result.get(
                "missing_inputs",
                implementability.get("missing_inputs", []),
            )
            verdict = {
                "status": "not_implementable",
                "scientific_claim_status": "not_evaluable",
                "hypothesis_id": hypothesis.get("id"),
                "interpretation": (
                    "The hypothesis was accepted as a scientific idea, but the current "
                    "representation/runner lacks the capabilities required to execute it."
                ),
                "missing_inputs": missing,
                "implementability": implementability,
                "unresolved_questions": [
                    "What capability upgrade is required to evaluate this hypothesis?",
                    "Can a lower-capability discriminating experiment test part of the hypothesis?",
                ],
            }
            record_hypothesis_outcome(hypothesis, verdict)
            return verdict

        # ---------------------------------------------------------
        # Diagnostic analysis: no CandidateModel/rollout required.
        # ---------------------------------------------------------
        if (
            isinstance(experiment_result, dict)
            and experiment_result.get("status") == "diagnostic_complete"
        ):
            analysis_type = experiment_result.get("analysis_type")
            summary = experiment_result.get("diagnostic_summary", {}) or {}
            global_metrics = experiment_result.get("global", {}) or {}
            bands = experiment_result.get("bands", []) or []

            if analysis_type != "operator_decomposition":
                verdict = {
                    "status": "inconclusive",
                    "scientific_claim_status": "not_evaluable",
                    "hypothesis_id": hypothesis.get("id"),
                    "interpretation": "Diagnostic completed with an unsupported analysis type for the verifier.",
                    "analysis_type": analysis_type,
                }
                record_hypothesis_outcome(hypothesis, verdict)
                return verdict

            skew_fraction = float(global_metrics.get("skew_fraction", 0.0))
            symmetric_fraction = float(global_metrics.get("symmetric_fraction", 0.0))
            skew_range = float(summary.get("bandwise_skew_range", 0.0))
            symmetric_range = float(summary.get("bandwise_symmetric_range", 0.0))
            reconstruction = float(
                global_metrics.get("symmetric_plus_skew_reconstruction_error", float("inf"))
            )

            # The decomposition identity is exact up to floating-point error.
            # The scientific claim is treated as supported only when both components
            # are demonstrably non-negligible and there is observable band heterogeneity.
            identity_ok = reconstruction <= 1e-10
            nontrivial_components = skew_fraction > 1e-3 and symmetric_fraction > 1e-3
            band_structure = skew_range > 1e-3 or symmetric_range > 1e-3

            if identity_ok and nontrivial_components and band_structure:
                status = "supported"
                interpretation = (
                    "The fitted development operator contains non-negligible symmetric and skew-symmetric "
                    "components, with observable variation of component fractions across the predefined POD bands. "
                    "This supports the presence of rotational-dissipative operator structure, but does not by itself "
                    "establish predictive superiority over the global linear baseline."
                )
            elif identity_ok and not nontrivial_components:
                status = "falsified"
                interpretation = (
                    "The fitted operator decomposition is algebraically valid, but one of the proposed "
                    "rotational or dissipative components is negligible at the diagnostic threshold."
                )
            else:
                status = "inconclusive"
                interpretation = (
                    "The diagnostic completed, but the evidence does not cleanly establish the proposed "
                    "rotational-dissipative band structure."
                )

            verdict = {
                "status": status,
                "scientific_claim_status": status,
                "hypothesis_id": hypothesis.get("id"),
                "analysis_type": analysis_type,
                "interpretation": interpretation,
                "diagnostic": {
                    "global_skew_fraction": skew_fraction,
                    "global_symmetric_fraction": symmetric_fraction,
                    "bandwise_skew_range": skew_range,
                    "bandwise_symmetric_range": symmetric_range,
                    "decomposition_identity_relative_error": reconstruction,
                    "identity_ok": identity_ok,
                    "nontrivial_components": nontrivial_components,
                    "band_structure_observed": band_structure,
                    "bands": bands,
                },
                "unresolved_questions": [
                    "Does the observed decomposition improve a predictive model when encoded structurally?",
                    "Are the band-wise differences stable under bootstrap or nearby train/validation windows?",
                    "Is the observed structure specific to POD coordinates or intrinsic to the flow dynamics?",
                ],
            }
            record_hypothesis_outcome(hypothesis, verdict)
            return verdict

        # ---------------------------------------------------------
        # Actual infrastructure/code execution failure.
        # ---------------------------------------------------------
        if not execution_ok or not experiment_result:
            verdict = {
                "status": "execution_failed",
                "scientific_claim_status": "not_evaluable",
                "hypothesis_id": hypothesis.get("id"),
                "interpretation": (
                    "The candidate could not be evaluated because the implementation "
                    "or execution pipeline failed. This is not scientific evidence against the hypothesis."
                ),
                "error": execution_error or "No experiment result available.",
                "unresolved_questions": [
                    "Can the hypothesis be evaluated after the implementation or execution failure is fixed?",
                    "Was the failure caused by the scientific mechanism or only by the execution infrastructure?",
                ],
            }
            record_hypothesis_outcome(hypothesis, verdict)
            return verdict

        one = self._find_one_step(experiment_result)
        rollout = self._find_rollout(experiment_result)

        if one is None:
            verdict = {
                "status": "inconclusive",
                "scientific_claim_status": "not_evaluable",
                "hypothesis_id": hypothesis.get("id"),
                "interpretation": (
                    "The experiment finished, but its output schema did not contain "
                    "comparable one-step metrics."
                ),
                "unresolved_questions": [
                    "How should this candidate's output be normalized for scientific comparison?",
                ],
            }
            record_hypothesis_outcome(hypothesis, verdict)
            return verdict

        candidate_one, global_one = one
        tolerance = 0.005
        one_step_ok = candidate_one <= global_one + tolerance
        meaningful_improvement = [
            row for row in rollout
            if row["candidate"] <= 0.99 * row["global"]
        ]
        any_improvement = [
            row for row in rollout
            if row["candidate"] < row["global"]
        ]

        if one_step_ok and meaningful_improvement:
            status = "supported"
            interpretation = (
                "The candidate stays within one-step tolerance and achieves at least one "
                "material rollout improvement over the global linear baseline."
            )
        elif candidate_one > global_one + tolerance:
            status = "falsified"
            interpretation = (
                "The candidate is materially worse than the global linear baseline already "
                "at one-step validation, so the proposed mechanism is not supported by the current evidence."
            )
        elif not any_improvement:
            status = "falsified"
            interpretation = (
                "The candidate improves none of the evaluated rollout horizons over the "
                "global linear baseline."
            )
        else:
            status = "inconclusive"
            interpretation = (
                "The candidate shows limited evidence of improvement, but not enough to "
                "satisfy the current confirmation threshold."
            )

        if status == "falsified":
            unresolved = [
                "Is the failed mechanism genuinely absent, or is the current representation insufficient?",
                "Could the observed local structure be finite-sample estimation noise?",
                "Is temporal memory a better explanation for the remaining rollout error?",
                "Would a different latent coordinate system expose structure that the current POD coordinates hide?",
            ]
        elif status == "supported":
            unresolved = [
                "Does the improvement persist under a confirmatory experiment?",
                "Is the mechanism distinct from known methods in the literature?",
                "Does the gain survive without validation-based hyperparameter tuning?",
            ]
        else:
            unresolved = [
                "Can a cleaner discriminating experiment separate the competing explanations?",
                "Is the observed gain large enough to justify additional complexity?",
            ]

        verdict = {
            "status": status,
            "scientific_claim_status": status,
            "hypothesis_id": hypothesis.get("id"),
            "interpretation": interpretation,
            "one_step": {
                "candidate": candidate_one,
                "global": global_one,
                "tolerance": tolerance,
                "margin": candidate_one - global_one,
            },
            "rollout": rollout,
            "improvement_horizons": [row["horizon"] for row in any_improvement],
            "meaningful_improvement_horizons": [row["horizon"] for row in meaningful_improvement],
            "unresolved_questions": unresolved,
        }
        record_hypothesis_outcome(hypothesis, verdict)
        return verdict

    def archive(
        self,
        hypothesis: Dict[str, Any],
        verdict: Dict[str, Any],
        cycle: int,
        experiment_result: Optional[Dict[str, Any]] = None,
        execution_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        hypothesis_id = str(hypothesis.get("id", "unknown")).replace("/", "_")
        out = self.root / "research" / "history" / f"cycle_{cycle:03d}_{hypothesis_id}"
        out.mkdir(parents=True, exist_ok=True)

        save_json(out / "hypothesis.json", hypothesis)
        save_json(out / "verdict.json", verdict)
        save_json(out / "experiment_result.json", experiment_result or {})
        save_json(out / "execution_steps.json", execution_steps or [])
        save_json(out / "execution_metadata.json", self._load_execution_metadata(self.root))

        sources = {
            "algorithm_spec.json": self.root / "algorithms/discovered/algorithm_spec.json",
            "algorithm_result.json": self.root / "algorithms/discovered/algorithm_result.json",
            "hypothesis_used.json": self.root / "algorithms/discovered/hypothesis_used.json",
            "algorithm_raw_response.txt": self.root / "research/candidates/algorithm_raw_response.txt",
            "code_result.json": self.root / "algorithms/discovered/implementation/code_result.json",
            "model.py": self.root / "algorithms/discovered/implementation/model.py",
            "model_raw_response.txt": self.root / "algorithms/discovered/implementation/model_raw_response.txt",
            "smoke_test.json": self.root / "experiments/runs/candidate_001/smoke_test.json",
            "development_result.json": self.root / "experiments/runs/candidate_001/development_result.json",
            "diagnostic_result.json": self.root / "experiments/runs/candidate_001/diagnostic_result.json",
        }
        for name, source in sources.items():
            if not source.exists():
                continue
            try:
                shutil.copy2(source, out / name)
            except OSError:
                pass

        self._attach_archive_path(hypothesis.get("id"), str(out))
        return out

    def _attach_archive_path(self, hypothesis_id: Optional[str], path: str) -> None:
        if not hypothesis_id:
            return
        state_path = self.root / "research" / "research_state.json"
        state = _load(state_path)
        if not isinstance(state, dict):
            return
        for item in state.get("hypothesis_history", []):
            if item.get("hypothesis_id") == hypothesis_id:
                item["artifact_dir"] = path
        save_json(state_path, state)


def _load(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
