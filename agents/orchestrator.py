"""Autonomous Research -> Algorithm -> Code -> Experiment -> Verify loop."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.research_agent import ResearchAgent
from agents.verifier_agent import VerifierAgent
from research.research_state import (
    HYPOTHESES_PATH,
    activate_hypothesis_for_execution,
    load_json,
    load_state,
    save_json,
    set_active_hypotheses,
)


class ScientificOrchestrator:
    def __init__(
        self,
        project_root: Optional[Path] = None,
        max_cycles: int = 3,
        hypotheses_per_cycle: int = 3,
    ):
        self.root = Path(project_root or Path(__file__).resolve().parents[1])
        self.max_cycles = max(1, int(max_cycles))
        self.hypotheses_per_cycle = max(1, int(hypotheses_per_cycle))
        self.research = ResearchAgent(project_root=self.root)
        self.verifier = VerifierAgent(project_root=self.root)

    def _run_script(self, name: str) -> Dict[str, Any]:
        path = self.root / "scripts" / name
        if not path.exists():
            return {
                "ok": False,
                "script": name,
                "error": f"Missing script: {path}",
            }

        process = subprocess.run(
            [sys.executable, str(path)],
            cwd=self.root,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "ok": process.returncode == 0,
            "script": name,
            "returncode": process.returncode,
            "stdout": process.stdout[-15000:],
            "stderr": process.stderr[-15000:],
        }

    def _cleanup_candidate_artifacts(self) -> None:
        """Remove stale candidate artifacts before evaluating a new hypothesis."""
        paths = [
            self.root / "algorithms/discovered/algorithm_result.json",
            self.root / "algorithms/discovered/algorithm_spec.json",
            self.root / "algorithms/discovered/algorithm_spec_raw.json",
            self.root / "algorithms/discovered/hypothesis_used.json",
            self.root / "algorithms/discovered/implementation/code_result.json",
            self.root / "algorithms/discovered/implementation/model.py",
            self.root / "experiments/runs/candidate_001/smoke_test.json",
            self.root / "experiments/runs/candidate_001/development_result.json",
            self.root / "experiments/runs/candidate_001/diagnostic_result.json",
        ]
        for path in paths:
            try:
                if path.exists() and path.is_file():
                    path.unlink()
            except OSError:
                pass

    def _activate(self, hypothesis: Dict[str, Any]) -> None:
        state = load_state()
        cycle = state.get("research_cycle", 0)
        active_ids = list(state.get("active_hypotheses", []))
        save_json(
            HYPOTHESES_PATH,
            {
                "status": "ok",
                "research_cycle": cycle,
                "hypotheses": [hypothesis],
                "active_hypothesis_id": hypothesis.get("id"),
                "all_cycle_hypothesis_ids": active_ids,
            },
        )
        activate_hypothesis_for_execution(hypothesis)

    def _restore(self, remaining_ids: List[str]) -> None:
        state = load_state()
        history = state.get("hypothesis_history", [])
        known = {item.get("hypothesis_id") for item in history}
        remaining_ids = [item for item in remaining_ids if item not in known]
        state["active_hypotheses"] = remaining_ids
        state["current_hypothesis"] = None
        state["status"] = "testing_cycle" if remaining_ids else "cycle_complete"
        save_json(self.root / "research" / "research_state.json", state)

    def _algorithm_result(self) -> Dict[str, Any]:
        value = load_json(self.root / "algorithms/discovered/algorithm_result.json", {})
        return value if isinstance(value, dict) else {}

    def _finalize_non_implementable(
        self,
        hypothesis: Dict[str, Any],
        cycle: int,
        execution_steps: List[Dict[str, Any]],
        algorithm_result: Dict[str, Any],
        remaining_ids: List[str],
    ) -> Dict[str, Any]:
        # Explicitly mark downstream stages as skipped, not failed.
        for skipped in ("run_code.py", "run_smoke_test.py", "run_experiment.py"):
            execution_steps.append({
                "ok": True,
                "script": skipped,
                "skipped": True,
                "reason": "algorithm_not_implementable",
            })

        verdict = self.verifier.verify(
            hypothesis=hypothesis,
            experiment_result={
                "status": "not_implementable",
                "candidate_id": algorithm_result.get("candidate_id"),
                "missing_inputs": algorithm_result.get("missing_inputs", []),
                "implementability": algorithm_result.get("implementability", {}),
            },
            execution_ok=True,
        )
        archive = self.verifier.archive(
            hypothesis=hypothesis,
            verdict=verdict,
            cycle=cycle,
            experiment_result=algorithm_result,
            execution_steps=execution_steps,
        )
        self._restore(remaining_ids)
        return {
            "hypothesis": hypothesis,
            "verdict": verdict,
            "execution_steps": execution_steps,
            "archive": str(archive),
        }

    def _finalize_diagnostic(
        self,
        hypothesis: Dict[str, Any],
        cycle: int,
        execution_steps: List[Dict[str, Any]],
        diagnostic_result: Dict[str, Any],
        remaining_ids: List[str],
    ) -> Dict[str, Any]:
        verdict = self.verifier.verify(
            hypothesis=hypothesis,
            experiment_result=diagnostic_result,
            execution_ok=True,
        )
        archive = self.verifier.archive(
            hypothesis=hypothesis,
            verdict=verdict,
            cycle=cycle,
            experiment_result=diagnostic_result,
            execution_steps=execution_steps,
        )
        self._restore(remaining_ids)
        return {
            "hypothesis": hypothesis,
            "verdict": verdict,
            "execution_steps": execution_steps,
            "archive": str(archive),
        }

    def _run_candidate(self, hypothesis: Dict[str, Any], cycle: int) -> Dict[str, Any]:
        state_before = load_state()
        cycle_remaining = list(state_before.get("active_hypotheses", []))
        self._cleanup_candidate_artifacts()
        self._activate(hypothesis)
        execution_steps: List[Dict[str, Any]] = []

        # Stage 1: Algorithm contract.
        step = self._run_script("run_algorithm.py")
        execution_steps.append(step)
        algorithm_result = self._algorithm_result()

        result_candidate_id = algorithm_result.get("candidate_id")
        result_hypothesis_id = algorithm_result.get("hypothesis_id")
        hypothesis_id = hypothesis.get("id")

        # Candidate IDs are normalized by AlgorithmAgent to the active Research
        # hypothesis ID. Accept legacy artifacts only when they explicitly
        # carry a matching hypothesis_id; never infer provenance from name.
        identity_ok = (
            result_candidate_id == hypothesis_id
            or result_hypothesis_id == hypothesis_id
        )
        if not identity_ok and (result_candidate_id or result_hypothesis_id):
            return self._execution_failure(
                hypothesis,
                cycle,
                execution_steps,
                (
                    "algorithm_result.json belongs to a different hypothesis; "
                    "refusing to continue with stale artifacts. "
                    f"expected={hypothesis_id}, candidate_id={result_candidate_id}, "
                    f"hypothesis_id={result_hypothesis_id}"
                ),
                cycle_remaining,
            )

        if algorithm_result.get("status") == "not_implementable":
            return self._finalize_non_implementable(
                hypothesis, cycle, execution_steps, algorithm_result, cycle_remaining
            )

        if not step["ok"]:
            error = (
                "run_algorithm.py failed.\n"
                f"STDOUT:\n{step.get('stdout', '')}\n"
                f"STDERR:\n{step.get('stderr', '')}"
            )
            return self._execution_failure(
                hypothesis, cycle, execution_steps, error, cycle_remaining
            )

        if algorithm_result.get("status") != "success":
            return self._execution_failure(
                hypothesis,
                cycle,
                execution_steps,
                f"Unexpected AlgorithmAgent status: {algorithm_result.get('status')}",
                cycle_remaining,
            )

        # Stage 2A: diagnostic analysis can execute without CandidateModel/code generation.
        algorithm_spec = load_json(
            self.root / "algorithms/discovered/algorithm_spec.json",
            {},
        )
        if isinstance(algorithm_spec, dict) and (
            algorithm_spec.get("execution_mode") == "analysis_runner"
            or algorithm_spec.get("candidate_type") == "diagnostic_analysis"
        ):
            step = self._run_script("run_diagnostic.py")
            execution_steps.append(step)
            diagnostic_result = load_json(
                self.root / "experiments/runs/candidate_001/diagnostic_result.json",
                {},
            )
            if isinstance(diagnostic_result, dict) and diagnostic_result.get("status") == "not_implementable":
                return self._finalize_non_implementable(
                    hypothesis,
                    cycle,
                    execution_steps,
                    diagnostic_result,
                    cycle_remaining,
                )
            if not step["ok"]:
                error = (
                    "run_diagnostic.py failed.\n"
                    f"STDOUT:\n{step.get('stdout', '')}\n"
                    f"STDERR:\n{step.get('stderr', '')}"
                )
                return self._execution_failure(
                    hypothesis, cycle, execution_steps, error, cycle_remaining
                )
            if not isinstance(diagnostic_result, dict) or diagnostic_result.get("status") != "diagnostic_complete":
                return self._execution_failure(
                    hypothesis,
                    cycle,
                    execution_steps,
                    f"Unexpected diagnostic status: {diagnostic_result.get('status') if isinstance(diagnostic_result, dict) else None}",
                    cycle_remaining,
                )
            return self._finalize_diagnostic(
                hypothesis, cycle, execution_steps, diagnostic_result, cycle_remaining
            )

        # Stage 2B: Code generation/validation.
        step = self._run_script("run_code.py")
        execution_steps.append(step)
        code_result = load_json(
            self.root / "algorithms/discovered/implementation/code_result.json",
            {},
        )

        # A code-stage capability mismatch is not an execution failure.
        if isinstance(code_result, dict) and code_result.get("status") == "not_implementable":
            for skipped in ("run_smoke_test.py", "run_experiment.py"):
                execution_steps.append({
                    "ok": True,
                    "script": skipped,
                    "skipped": True,
                    "reason": "code_stage_not_implementable",
                })
            verdict = self.verifier.verify(
                hypothesis=hypothesis,
                experiment_result=code_result,
                execution_ok=True,
            )
            archive = self.verifier.archive(
                hypothesis=hypothesis,
                verdict=verdict,
                cycle=cycle,
                experiment_result=code_result,
                execution_steps=execution_steps,
            )
            self._restore(cycle_remaining)
            return {
                "hypothesis": hypothesis,
                "verdict": verdict,
                "execution_steps": execution_steps,
                "archive": str(archive),
            }

        if not step["ok"]:
            error = (
                "run_code.py failed.\n"
                f"STDOUT:\n{step.get('stdout', '')}\n"
                f"STDERR:\n{step.get('stderr', '')}"
            )
            return self._execution_failure(
                hypothesis, cycle, execution_steps, error, cycle_remaining
            )

        if not isinstance(code_result, dict) or code_result.get("status") != "success":
            return self._execution_failure(
                hypothesis,
                cycle,
                execution_steps,
                f"Unexpected CodeAgent status: {code_result.get('status') if isinstance(code_result, dict) else None}",
                cycle_remaining,
            )

        # Stage 3-4: Smoke test and development experiment.
        for script in ("run_smoke_test.py", "run_experiment.py"):
            step = self._run_script(script)
            execution_steps.append(step)
            if not step["ok"]:
                error = (
                    f"{script} failed.\n"
                    f"STDOUT:\n{step.get('stdout', '')}\n"
                    f"STDERR:\n{step.get('stderr', '')}"
                )
                return self._execution_failure(
                    hypothesis, cycle, execution_steps, error, cycle_remaining
                )

        result = load_json(
            self.root / "experiments/runs/candidate_001/development_result.json",
            None,
        )
        verdict = self.verifier.verify(
            hypothesis=hypothesis,
            experiment_result=result,
            execution_ok=True,
        )
        archive = self.verifier.archive(
            hypothesis=hypothesis,
            verdict=verdict,
            cycle=cycle,
            experiment_result=result,
            execution_steps=execution_steps,
        )
        self._restore(cycle_remaining)
        return {
            "hypothesis": hypothesis,
            "verdict": verdict,
            "execution_steps": execution_steps,
            "archive": str(archive),
        }

    def _execution_failure(
        self,
        hypothesis: Dict[str, Any],
        cycle: int,
        execution_steps: List[Dict[str, Any]],
        error: str,
        remaining_ids: List[str],
    ) -> Dict[str, Any]:
        verdict = self.verifier.verify(
            hypothesis=hypothesis,
            experiment_result=None,
            execution_ok=False,
            execution_error=error,
        )
        archive = self.verifier.archive(
            hypothesis=hypothesis,
            verdict=verdict,
            cycle=cycle,
            experiment_result=None,
            execution_steps=execution_steps,
        )
        self._restore(remaining_ids)
        return {
            "hypothesis": hypothesis,
            "verdict": verdict,
            "execution_steps": execution_steps,
            "archive": str(archive),
        }

    def run(self) -> Dict[str, Any]:
        all_cycles: List[Dict[str, Any]] = []
        diagnostics: Dict[str, Any] = {}

        for _ in range(self.max_cycles):
            print("\n" + "=" * 78)
            print("AUTONOMOUS RESEARCH CYCLE")
            print("=" * 78)

            try:
                research_result = self.research.discover_next(
                    diagnostics=diagnostics,
                    max_hypotheses=self.hypotheses_per_cycle,
                )
            except Exception as exc:
                print(f"[ResearchAgent] DISCOVERY BLOCKED: {exc}")
                break

            cycle = research_result["research_cycle"]
            hypotheses = research_result["hypotheses"]
            set_active_hypotheses(hypotheses, research_result.get("strategy"))

            cycle_result: Dict[str, Any] = {
                "cycle": cycle,
                "research": research_result,
                "candidates": [],
            }

            print(f"[ResearchAgent] Generated {len(hypotheses)} hypotheses.")
            for hypothesis in hypotheses:
                print(f"  {hypothesis.get('id')}: {hypothesis.get('name')}")

            for hypothesis in hypotheses:
                candidate_result = self._run_candidate(hypothesis, cycle)
                cycle_result["candidates"].append(candidate_result)
                diagnostics = {
                    "latest_verdict": candidate_result.get("verdict", {}),
                    "execution_steps": candidate_result.get("execution_steps", []),
                }
                verdict = candidate_result.get("verdict", {})
                print(
                    f"[Verifier] {hypothesis.get('id')} -> "
                    f"{verdict.get('status')}"
                )

            summary_path = (
                self.root / "research" / "discovery_cycles" / f"cycle_{cycle:03d}_summary.json"
            )
            save_json(summary_path, cycle_result)
            all_cycles.append(cycle_result)

        return {"cycles": all_cycles}
