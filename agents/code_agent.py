from __future__ import annotations

import json
import os
import re
from pathlib import Path
from textwrap import dedent
from typing import Any

from tools.minimax_client import MiniMaxClient
from runtime.algorithm_contract import ContractValidationError, validate_algorithm_contract
from runtime.code_validator import CodeValidationError, save_validated_code

ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATH = ROOT / "runtime" / "capability_profile.json"


def load_capability_profile() -> dict[str, Any]:
    if CAPABILITY_PATH.exists():
        return json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))
    return {
        "representation": {"type": "POD_latent", "latent_dimension": 52},
        "available_inputs": ["latent_state", "latent_history"],
        "unavailable_inputs": ["exogenous_parameters", "velocity_components"],
        "runner": {"candidate_interface": "first_order_state", "explicit_history_input": False},
    }


class CodeAgent:
    """Implement an approved scientific contract without mechanism substitution."""

    def __init__(self, model: str = "MiniMax-M3") -> None:
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise RuntimeError("MINIMAX_API_KEY is not set")
        self.client = MiniMaxClient(
            api_key=api_key,
            base_url=os.getenv("MINIMAX_BASE_URL"),
            model=model,
            connect_timeout=15.0,
            read_timeout=180.0,
            max_retries=3,
        )
        self.model = model

    @staticmethod
    def save_json(data: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_contract(self, spec_path: Path) -> dict[str, Any]:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        try:
            validate_algorithm_contract(spec)
        except ContractValidationError as exc:
            raise ContractValidationError(str(exc)) from exc
        return spec

    @staticmethod
    def contract_is_executable(spec: dict[str, Any]) -> tuple[bool, list[str]]:
        capability = load_capability_profile()
        impl = spec.get("implementability", {})
        if impl.get("current_pipeline_compatible") is False:
            return False, list(impl.get("missing_inputs", []))

        req = spec.get("implementation_requirements", {})
        available = set(capability.get("available_inputs", []))
        missing = []
        for item in req.get("required_inputs", []):
            if item not in available:
                missing.append(item)
        for item in req.get("required_external_parameters", []):
            if item not in available:
                missing.append(item)
        for item in req.get("required_field_components", []):
            if item not in available:
                missing.append(item)
        history = int(req.get("required_history", 0))
        if history > 0 and not capability.get("runner", {}).get("explicit_history_input", False):
            missing.append("runner_explicit_history_input")
        return len(set(missing)) == 0, sorted(set(missing))

    @staticmethod
    def build_system_prompt() -> str:
        return dedent("""
        You are the Code Agent of an autonomous scientific discovery harness.
        Implement the approved Algorithm Contract exactly. Do not replace the scientific mechanism.
        Output only Python code defining class CandidateModel.
        Required public methods:
          __init__(latent_dim, seed=0)
          fit(z_train)
          predict_next(z_t)
          rollout(z0, horizon)
        Allowed imports: numpy, math, typing, dataclasses.
        No filesystem, network, shell, subprocess, eval, or exec.
        If the contract cannot be faithfully represented by this interface, do not invent inputs.
        """).strip()

    @staticmethod
    def build_user_prompt(spec: dict[str, Any]) -> str:
        return (
            "Implement this approved contract exactly. Do not alter the mathematical mechanism.\n\n"
            + json.dumps(spec, ensure_ascii=False, indent=2)
            + "\n\nReturn only Python code."
        )

    def generate_code(self, spec: dict[str, Any]) -> str:
        return self.client.chat([
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(spec)},
        ], temperature=0.1)

    @staticmethod
    def clean_code(code: str) -> str:
        code = code.strip()
        code = re.sub(r"^```python\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
        return code.strip()

    @staticmethod
    def fallback_supported(spec: dict[str, Any]) -> bool:
        dynamics_type = str(spec.get("dynamics", {}).get("type", "")).lower()
        req = spec.get("implementation_requirements", {})
        allowed = {
            "linear_operator",
            "residual_operator",
            "global_linear",
            "shrinkage_linear_operator",
        }
        return (
            dynamics_type in allowed
            and not req.get("required_external_parameters")
            and not req.get("required_field_components")
            and int(req.get("required_history", 0)) == 0
        )

    @staticmethod
    def build_deterministic_model_code(spec: dict[str, Any]) -> str:
        latent_dim = int(spec["representation"]["latent_dimension"])
        dynamics_type = str(spec.get("dynamics", {}).get("type", "linear_operator")).lower()
        residual = dynamics_type == "residual_operator"
        ridge = 1e-5 if dynamics_type == "shrinkage_linear_operator" else 1e-8
        return dedent(f"""
        import numpy as np

        class CandidateModel:
            def __init__(self, latent_dim: int, seed: int = 0):
                self.latent_dim = int(latent_dim)
                self.seed = int(seed)
                if self.latent_dim != {latent_dim}:
                    raise ValueError("latent_dim does not match approved contract")
                self.ridge = {ridge!r}
                self.operator = None
                self.fitted = False

            def fit(self, z_train):
                z = np.asarray(z_train, dtype=float)
                if z.ndim != 2 or z.shape[1] != self.latent_dim:
                    raise ValueError("z_train must have shape (n_samples, latent_dim)")
                if z.shape[0] < 2:
                    raise ValueError("z_train requires at least two states")
                X = z[:-1]
                Y = z[1:]
                {"Y = Y - X" if residual else ""}
                gram = X.T @ X
                scale = max(float(np.trace(gram) / max(self.latent_dim, 1)), 1.0)
                reg = self.ridge * scale * np.eye(self.latent_dim)
                coef = np.linalg.solve(gram + reg, X.T @ Y)
                self.operator = coef.T
                self.fitted = True
                return self

            def predict_next(self, z_t):
                if not self.fitted:
                    raise RuntimeError("model has not been fitted")
                z = np.asarray(z_t, dtype=float).reshape(-1)
                if z.size != self.latent_dim:
                    raise ValueError("z_t has incorrect dimension")
                out = self.operator @ z
                {"out = z + out" if residual else ""}
                return out

            def rollout(self, z0, horizon):
                if int(horizon) < 0:
                    raise ValueError("horizon must be non-negative")
                z0 = np.asarray(z0, dtype=float).reshape(-1)
                if z0.size != self.latent_dim:
                    raise ValueError("z0 has incorrect dimension")
                traj = np.zeros((int(horizon) + 1, self.latent_dim), dtype=float)
                traj[0] = z0
                cur = z0.copy()
                for k in range(int(horizon)):
                    cur = self.predict_next(cur)
                    traj[k + 1] = cur
                if not np.all(np.isfinite(traj)):
                    raise FloatingPointError("rollout contains NaN or Inf")
                return traj
        """).strip() + "\n"

    @staticmethod
    def validate_generated_code(code: str, model_path: Path) -> dict[str, Any]:
        return save_validated_code(code, model_path)

    def run(self, spec_path: Path, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            spec = self.load_contract(Path(spec_path))
        except Exception as exc:
            result = {"status": "contract_load_failed", "error": str(exc)}
            self.save_json(result, output_dir / "code_result.json")
            return result

        executable, missing = self.contract_is_executable(spec)
        candidate_id = spec.get("candidate_id")
        candidate_name = spec.get("name")

        if not executable:
            result = {
                "status": "not_implementable",
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "missing_inputs": missing,
                "implementability": spec.get("implementability", {}),
            }
            self.save_json(result, output_dir / "code_result.json")
            return result

        model_path = output_dir / "model.py"
        llm_error = None
        llm_code = None

        try:
            llm_code = self.clean_code(self.generate_code(spec))
            validation = self.validate_generated_code(llm_code, model_path)
            result = {
                "status": "success",
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "code_source": "llm",
                "model_path": str(model_path),
                "validation": validation,
            }
            self.save_json(result, output_dir / "code_result.json")
            return result
        except Exception as exc:
            llm_error = str(exc)
            if model_path.exists():
                model_path.unlink()

        if not self.fallback_supported(spec):
            result = {
                "status": "code_generation_failed",
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "error": "LLM implementation failed and no faithful deterministic fallback exists.",
                "llm_error": llm_error,
            }
            self.save_json(result, output_dir / "code_result.json")
            return result

        try:
            fallback = self.build_deterministic_model_code(spec)
            validation = self.validate_generated_code(fallback, model_path)
            result = {
                "status": "success",
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "code_source": "deterministic_fallback",
                "model_path": str(model_path),
                "validation": validation,
                "llm_error": llm_error,
            }
            self.save_json(result, output_dir / "code_result.json")
            return result
        except Exception as exc:
            result = {
                "status": "code_validation_failed",
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "error": str(exc),
                "llm_error": llm_error,
            }
            self.save_json(result, output_dir / "code_result.json")
            return result
