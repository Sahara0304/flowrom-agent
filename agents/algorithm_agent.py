from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any

from tools.minimax_client import MiniMaxClient

from runtime.algorithm_contract import (
    ContractValidationError,
    save_validated_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATH = ROOT / "runtime" / "capability_profile.json"


def load_capability_profile() -> dict[str, Any]:
    if not CAPABILITY_PATH.exists():
        return {
            "version": 1,
            "representation": {
                "type": "POD_latent",
                "latent_dimension": 52,
                "source_field": "magnitude",
            },
            "available_inputs": [
                "latent_state",
                "latent_history",
            ],
            "unavailable_inputs": [
                "pressure",
                "velocity_components",
                "divergence",
                "curl",
                "boundary_conditions",
                "exogenous_parameters",
                "external_control_inputs",
            ],
            "supported_dynamics": [
                "linear_operator",
                "residual_operator",
                "autoregressive",
            ],
            "runner": {
                "candidate_interface": "first_order_state",
                "explicit_history_input": False,
                "latent_dimension": 52,
            },
        }

    with CAPABILITY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


class AlgorithmAgent:
    """
    Research Hypothesis
            ↓
    Algorithm Contract
            ↓
    Scientific/Capability checks
            ↓
    Validated Contract

    Important:
    - The agent never invents Candidate 002/003/etc.
    - The Research Agent supplies the hypothesis.
    - The contract is normalized against the actual execution capability.
    - A scientifically interesting but currently unexecutable hypothesis is
      returned as `not_implementable`; it is not silently converted into
      another algorithm.
    """

    def __init__(self, model: str = "MiniMax-M3"):
        api_key = os.getenv("MINIMAX_API_KEY")

        if not api_key:
            raise RuntimeError(
                "没有找到环境变量 MINIMAX_API_KEY。\n"
                '请先执行：$env:MINIMAX_API_KEY="..."'
            )

        self.client = MiniMaxClient(
            api_key=api_key,
            model=model,
            connect_timeout=15.0,
            read_timeout=180.0,
            max_retries=3,
        )

        self.model = model

    # ========================================================
    # JSON
    # ========================================================

    @staticmethod
    def save_json(data: Any, path: Path) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def _clean_json_text(text: str) -> str:
        text = text.strip()
        text = text.lstrip("\ufeff")

        text = re.sub(
            r"^\s*```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```\s*$",
            "",
            text,
        )

        replacements = {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(
            r",\s*([}\]])",
            r"\1",
            text,
        )

        return text.strip()

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escape = False

        for index in range(
            start,
            len(text),
        ):
            char = text[index]

            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

                if depth == 0:
                    return text[
                        start:index + 1
                    ]

        return None

    def _parse_json(
        self,
        text: str,
        allow_repair: bool = True,
    ) -> dict[str, Any]:

        if not text:
            raise ValueError(
                "LLM returned empty response."
            )

        cleaned = self._clean_json_text(text)

        candidates = [cleaned]

        extracted = self._extract_json_object(
            cleaned
        )

        if extracted:
            candidates.append(
                self._clean_json_text(
                    extracted
                )
            )

        for candidate in candidates:
            try:
                value = json.loads(candidate)

                if isinstance(value, dict):
                    return value

            except json.JSONDecodeError:
                pass

            try:
                value = ast.literal_eval(
                    candidate
                )

                if isinstance(value, dict):
                    return dict(value)

            except (
                SyntaxError,
                ValueError,
                TypeError,
            ):
                pass

        if allow_repair:
            return self._repair_json_with_llm(
                text
            )

        raise ValueError(
            "Unable to parse valid JSON."
        )

    def _repair_json_with_llm(
        self,
        raw_text: str,
    ) -> dict[str, Any]:

        prompt = f"""
你是 Algorithm Contract JSON Repair Agent。

下面内容应该是 JSON object，但可能存在：
- Markdown
- 多余解释
- 单引号
- trailing comma
- 非法反斜杠
- 未转义换行

你的任务只有一个：
把它修复成严格合法的 JSON object。

禁止改变：
- algorithm mechanism
- mathematical meaning
- candidate identity
- latent dimension
- scientific claim

禁止新增任何科学内容。

不要输出 Markdown。
不要输出解释。
只输出 JSON object。

原始内容：

{raw_text[:40000]}
""".strip()

        content = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You repair JSON only. "
                        "Return JSON and nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
        )

        if not content:
            raise RuntimeError(
                "JSON repair returned empty content."
            )

        return self._parse_json(
            content,
            allow_repair=False,
        )

    # ========================================================
    # Research loading
    # ========================================================

    def load_research(
        self,
        research_dir: Path,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
    ]:

        hypothesis_path = (
            research_dir
            / "research_hypotheses.json"
        )

        state_path = (
            research_dir
            / "research_state.json"
        )

        if not hypothesis_path.exists():
            raise FileNotFoundError(
                f"找不到：{hypothesis_path}"
            )

        if not state_path.exists():
            raise FileNotFoundError(
                f"找不到：{state_path}"
            )

        with hypothesis_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            hypotheses = json.load(f)

        with state_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            state = json.load(f)

        return hypotheses, state

    def _load_active_hypothesis(
        self,
        research_dir: Path,
        hypotheses: dict[str, Any],
    ) -> dict[str, Any]:

        active_path = (
            research_dir
            / "active_hypothesis.json"
        )

        active_id = None

        if active_path.exists():
            try:
                with active_path.open(
                    "r",
                    encoding="utf-8",
                ) as f:
                    active = json.load(f)

                active_id = active.get(
                    "id"
                )

            except (
                OSError,
                json.JSONDecodeError,
            ):
                active_id = None

        candidates = hypotheses.get(
            "hypotheses",
            [],
        )

        if not isinstance(
            candidates,
            list,
        ):
            return {}

        # Exact active hypothesis match first.
        if active_id:
            for candidate in candidates:
                if (
                    isinstance(candidate, dict)
                    and
                    candidate.get("id")
                    == active_id
                ):
                    return candidate

        # Fallback only when there is exactly one candidate.
        valid = [
            item
            for item in candidates
            if self._valid_hypothesis(item)
        ]

        if len(valid) == 1:
            return valid[0]

        # Never silently choose the first candidate from a
        # multi-hypothesis ResearchAgent result.
        return {}

    @staticmethod
    def _valid_hypothesis(
        hypothesis: Any,
    ) -> bool:

        if not isinstance(
            hypothesis,
            dict,
        ):
            return False

        return bool(
            str(
                hypothesis.get(
                    "name",
                    "",
                )
            ).strip()
        )

    # ========================================================
    # Capability analysis
    # ========================================================

    @staticmethod
    def _contains_any(
        text: str,
        tokens: tuple[str, ...],
    ) -> bool:

        text = text.lower()

        return any(
            token in text
            for token in tokens
        )

    @staticmethod
    def _canonical_dynamics_type(
        hypothesis: dict[str, Any],
        spec: dict[str, Any],
    ) -> str:
        """Map free-form LLM dynamics labels to a small executable taxonomy."""
        dyn = spec.get("dynamics", {}) or {}
        raw = str(dyn.get("type", "")).strip().lower()
        combined = " ".join(
            [
                str(hypothesis.get("name", "")),
                str(hypothesis.get("mechanism", "")),
                str(hypothesis.get("mathematical_form", "")),
                str(spec.get("name", "")),
                str(spec.get("scientific_claim", "")),
                str(dyn.get("equation", "")),
                raw,
            ]
        ).lower()

        # Strong mechanism keywords take precedence over vague LLM labels.
        if any(k in combined for k in (
            "mixture-of-experts",
            "mixture of experts",
            "state-conditioned",
            "state conditioned",
            "state-dependent operator",
            "state dependent operator",
            "radial-basis",
            "radial basis",
            "partition of unity",
            "softmax gating",
            "operator interpolation",
        )):
            return "state_dependent_operator"
        if any(k in combined for k in (
            "quadratic-bilinear",
            "quadratic bilinear",
            "quadratic-bilinear coupling",
            "polynomial quadratic",
        )):
            return "quadratic_bilinear"
        if any(k in combined for k in ("residual", "deltaa", "delta a", "closure correction")):
            return "residual_operator"
        if any(k in combined for k in ("shrinkage", "ridge-regularized", "ridge regularized")):
            return "shrinkage_linear_operator"
        if any(k in combined for k in ("autoregressive", "second-order", "third-order", "memory", "history", "delay")):
            return "autoregressive"
        if any(k in combined for k in ("lifting", "koopman", "feature lift", "nonlinear feature", "diffusion map")):
            return "nonlinear_lifting"
        if any(k in combined for k in ("graph-structured", "graph structured", "block partition", "sparse inter-block")):
            return "graph_structured"
        if any(k in combined for k in ("gaussian process", "reproducing kernel", "rkhs", "kernel operator")):
            return "kernel_operator"
        if any(k in combined for k in ("sde", "stochastic ito", "diffusion process", "state-dependent diffusion")):
            return "stochastic_sde"
        if any(k in combined for k in ("port-hamiltonian", "hamiltonian", "symplectic")):
            return "port_hamiltonian"
        if any(k in combined for k in ("global linear", "global operator", "linear operator")):
            return "global_linear"
        return raw or "unknown"

    @staticmethod
    def _detect_experiment_plan(
        hypothesis: dict[str, Any],
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Classify whether the hypothesis is a model experiment or an analysis experiment.

        The first diagnostic experiment intentionally supported by the current harness is
        an operator symmetric/skew decomposition.  It only requires the existing 52-D POD
        latent trajectory; it does not require a new predictor interface.
        """
        text = " ".join(
            [
                str(hypothesis.get("name", "")),
                str(hypothesis.get("scientific_question", "")),
                str(hypothesis.get("mechanism", "")),
                str(hypothesis.get("mathematical_form", "")),
                str(spec.get("name", "")),
                str(spec.get("scientific_claim", "")),
                str(spec.get("dynamics", {}).get("equation", "")),
                str(spec.get("dynamics", {}).get("operator_structure", "")),
            ]
        ).lower()

        if any(
            token in text
            for token in (
                "rotational-dissipative",
                "rotational dissipative",
                "skew/symmetric",
                "skew symmetric",
                "skew-symmetric",
                "symmetric/skew",
                "symmetric and skew",
                "operator decomposition",
                "spectral decomposition",
                "band-wise",
                "band wise",
            )
        ):
            return {
                "candidate_type": "diagnostic_analysis",
                "execution_mode": "analysis_runner",
                "experiment_type": "diagnostic_analysis",
                "analysis_type": "operator_decomposition",
            }

        # Respect an explicit experiment declaration from the LLM when present.
        explicit = spec.get("experiment", {})
        if isinstance(explicit, dict):
            analysis_type = str(explicit.get("analysis_type", "")).strip()
            experiment_type = str(explicit.get("type", "")).strip()
            if experiment_type or analysis_type:
                return {
                    "candidate_type": spec.get("candidate_type", "predictive_model"),
                    "execution_mode": spec.get("execution_mode", "model_runner"),
                    "experiment_type": experiment_type or "predictive_model",
                    "analysis_type": analysis_type or None,
                }

        return {
            "candidate_type": "predictive_model",
            "execution_mode": "model_runner",
            "experiment_type": "predictive_model",
            "analysis_type": None,
        }

    def _infer_requirements(
        self,
        hypothesis: dict[str, Any],
        spec: dict[str, Any],
    ) -> dict[str, Any]:

        capability = (
            load_capability_profile()
        )

        experiment = self._detect_experiment_plan(hypothesis, spec)
        spec["candidate_type"] = experiment["candidate_type"]
        spec["execution_mode"] = experiment["execution_mode"]
        spec["experiment"] = {
            "type": experiment["experiment_type"],
            "analysis_type": experiment.get("analysis_type"),
        }

        rep = capability.get(
            "representation",
            {},
        )

        combined = " ".join(
            [
                str(
                    hypothesis.get(
                        "name",
                        "",
                    )
                ),
                str(
                    hypothesis.get(
                        "mechanism",
                        "",
                    )
                ),
                str(
                    hypothesis.get(
                        "mathematical_form",
                        "",
                    )
                ),
                str(
                    spec.get(
                        "name",
                        "",
                    )
                ),
                str(
                    spec.get(
                        "scientific_claim",
                        "",
                    )
                ),
                str(
                    spec.get(
                        "dynamics",
                        {},
                    ).get(
                        "equation",
                        "",
                    )
                ),
            ]
        ).lower()

        dynamics = spec.setdefault(
            "dynamics",
            {},
        )

        requirements = spec.setdefault(
            "implementation_requirements",
            {},
        )

        requirements.setdefault(
            "required_inputs",
            [
                "latent_state",
            ],
        )

        requirements.setdefault(
            "required_history",
            0,
        )

        requirements.setdefault(
            "required_external_parameters",
            [],
        )

        requirements.setdefault(
            "required_field_components",
            [],
        )

        requirements.setdefault(
            "required_representation",
            rep.get(
                "type",
                "POD_latent",
            ),
        )

        requirements.setdefault(
            "required_latent_dimension",
            rep.get(
                "latent_dimension",
                52,
            ),
        )

        # The LLM may over-specify capabilities in the contract. Only infer
        # external parameters/history when the Research hypothesis itself
        # explicitly requires them. Endogenous state dependence (e.g.
        # state-conditioned gating, mixture-of-experts, RBF partitions) is
        # not an exogenous input and does not require explicit history.
        hypothesis_text = " ".join(
            [
                str(hypothesis.get("name", "")),
                str(hypothesis.get("mechanism", "")),
                str(hypothesis.get("mathematical_form", "")),
                str(hypothesis.get("scientific_question", "")),
            ]
        ).lower()

        explicit_history = self._contains_any(
            hypothesis_text,
            (
                "non-markov",
                "nonmarkov",
                "delay-coordinate",
                "delay coordinate",
                "memory",
                "history",
                "autoregressive",
                "lagged state",
                "z(t-1)",
                "z_{t-1}",
                "second-order",
                "third-order",
                "higher-order",
            ),
        )

        explicit_external = self._contains_any(
            hypothesis_text,
            (
                "exogenous parameter",
                "exogenous input",
                "external parameter",
                "external input",
                "control input",
                "control signal",
                "parameter-conditioned",
                "conditioned on parameter",
            ),
        )

        if not explicit_history:
            requirements["required_history"] = 0

        if not explicit_external:
            requirements["required_external_parameters"] = []

        # Delay / memory models.
        if explicit_history:
            inferred_history = max(
                1,
                int(
                    requirements.get(
                        "required_history",
                        0,
                    )
                ),
            )

            if "second-order" in combined:
                inferred_history = max(
                    inferred_history,
                    1,
                )

            if "third-order" in combined:
                inferred_history = max(
                    inferred_history,
                    2,
                )

            requirements[
                "required_history"
            ] = inferred_history

        # Exogenous parameters.
        if explicit_external:
            current = list(
                requirements.get(
                    "required_external_parameters",
                    [],
                )
            )

            if "exogenous_parameter" not in current:
                current.append(
                    "exogenous_parameter"
                )

            requirements[
                "required_external_parameters"
            ] = current

        # Vector field / divergence constrained models.
        if self._contains_any(
            combined,
            (
                "divergence-free",
                "divergence free",
                "divergence-free",
                "velocity components",
                "vector field",
                "symplectic",
                "hamiltonian",
            ),
        ):
            current = list(
                requirements.get(
                    "required_field_components",
                    [],
                )
            )

            if (
                "velocity_components"
                not in current
            ):
                current.append(
                    "velocity_components"
                )

            requirements[
                "required_field_components"
            ] = current

        # Fix latent dimension to actual capability.
        requirements[
            "required_latent_dimension"
        ] = int(
            rep.get(
                "latent_dimension",
                52,
            )
        )

        # The current smoke/experiment runner provides only
        # predict_next(z_t), not explicit history or external
        # conditioning. Mark those candidates unexecutable
        # rather than faking them.
        missing = []

        available_inputs = set(
            capability.get(
                "available_inputs",
                [],
            )
        )

        unavailable_inputs = set(
            capability.get(
                "unavailable_inputs",
                [],
            )
        )

        for item in requirements.get(
            "required_inputs",
            [],
        ):
            if item not in available_inputs:
                missing.append(item)

        for item in requirements.get(
            "required_external_parameters",
            [],
        ):
            if item not in available_inputs:
                missing.append(item)

        for item in requirements.get(
            "required_field_components",
            [],
        ):
            if item not in available_inputs:
                missing.append(item)

        runner = capability.get(
            "runner",
            {},
        )

        if (
            int(
                requirements.get(
                    "required_history",
                    0,
                )
            )
            > 0
            and
            not runner.get(
                "explicit_history_input",
                False,
            )
        ):
            missing.append(
                "runner_explicit_history_input"
            )

        required_rep = (
            requirements.get(
                "required_representation"
            )
        )

        current_rep = (
            rep.get(
                "type"
            )
        )

        if (
            required_rep
            !=
            current_rep
        ):
            missing.append(
                f"representation:{required_rep}"
            )

        required_dim = int(
            requirements.get(
                "required_latent_dimension",
                52,
            )
        )

        # Normalize the free-form LLM label to the capability taxonomy.
        canonical_type = self._canonical_dynamics_type(
            hypothesis,
            spec,
        )
        dynamics["type"] = canonical_type

        supported_dynamics = {
            str(item).strip().lower()
            for item in capability.get("supported_dynamics", [])
        }

        supported_experiments = {
            str(item).strip().lower()
            for item in capability.get("supported_experiment_types", [])
        }
        supported_analyses = {
            str(item).strip().lower()
            for item in capability.get("supported_analyses", [])
        }

        experiment_type = str(spec.get("experiment", {}).get("type", "predictive_model")).lower()
        analysis_type = str(spec.get("experiment", {}).get("analysis_type", "") or "").lower()

        if experiment_type == "diagnostic_analysis":
            if experiment_type not in supported_experiments:
                missing.append(f"experiment_type:{experiment_type}")
            if analysis_type and analysis_type not in supported_analyses:
                missing.append(f"analysis_type:{analysis_type}")
        elif canonical_type not in supported_dynamics:
            missing.append(f"dynamics:{canonical_type}")

        current_dim = int(
            rep.get(
                "latent_dimension",
                52,
            )
        )

        if required_dim != current_dim:
            missing.append(
                f"latent_dimension:{required_dim}"
            )

        spec[
            "implementability"
        ] = {
            "current_pipeline_compatible":
                len(set(missing)) == 0,
            "missing_inputs":
                sorted(set(missing)),
            "required_history":
                int(
                    requirements.get(
                        "required_history",
                        0,
                    )
                ),
            "required_representation":
                required_rep,
            "required_latent_dimension":
                required_dim,
            "dynamics_type":
                canonical_type,
            "supported_dynamics":
                sorted(supported_dynamics),
            "experiment_type": experiment_type,
            "analysis_type": analysis_type or None,
            "supported_experiment_types": sorted(supported_experiments),
            "supported_analyses": sorted(supported_analyses),
        }

        return requirements

    # ========================================================
    # Contract normalization
    # ========================================================

    def _normalize_spec(
        self,
        spec: dict[str, Any],
        hypothesis: dict[str, Any],
    ) -> dict[str, Any]:

        capability = (
            load_capability_profile()
        )

        rep_cap = capability.get(
            "representation",
            {},
        )

        # Candidate identity belongs to the Research hypothesis, not to the
        # LLM-generated contract. Preserve the LLM suggestion for provenance,
        # but force the executable contract to use the active hypothesis id.
        generated_candidate_id = spec.get("candidate_id")
        spec["generated_candidate_id"] = generated_candidate_id
        spec["candidate_id"] = hypothesis.get(
            "id",
            "unknown_candidate",
        )
        spec["hypothesis_id"] = hypothesis.get(
            "id",
            "unknown_hypothesis",
        )

        spec.setdefault(
            "name",
            hypothesis.get(
                "name",
                "Unnamed candidate",
            ),
        )

        spec.setdefault(
            "paradigm",
            "classical",
        )

        h = spec.setdefault(
            "hypothesis",
            {},
        )

        h.setdefault(
            "source",
            "research_agent",
        )

        h.setdefault(
            "motivation",
            hypothesis.get(
                "motivation",
                "",
            ),
        )

        h.setdefault(
            "novelty_status",
            "verification_incomplete",
        )

        representation = spec.setdefault(
            "representation",
            {},
        )

        # Always use the real latent dimension.
        representation[
            "latent_dimension"
        ] = int(
            rep_cap.get(
                "latent_dimension",
                52,
            )
        )

        representation[
            "input_state"
        ] = representation.get(
            "input_state",
            "z_t",
        )

        # Do not let the LLM put LaTeX/final-test narrative
        # into the normalization field.
        representation[
            "normalization"
        ] = (
            "POD latent coefficients from development training data; "
            "no additional normalization."
        )

        dynamics = spec.setdefault(
            "dynamics",
            {},
        )

        dynamics.setdefault(
            "equation",
            hypothesis.get(
                "mathematical_form",
                "z_next = A @ z_t",
            ),
        )

        dynamics.setdefault(
            "inputs",
            [
                "z_t: latent state",
            ],
        )

        dynamics.setdefault(
            "outputs",
            [
                "z_next: predicted latent state",
            ],
        )

        dynamics.setdefault(
            "parameters",
            [],
        )

        dynamics.setdefault(
            "operator_structure",
            "",
        )

        dynamics.setdefault(
            "stability_mechanism",
            "",
        )

        experiment = spec.setdefault("experiment", {})
        if not isinstance(experiment, dict):
            experiment = {}
            spec["experiment"] = experiment
        detected = self._detect_experiment_plan(hypothesis, spec)
        spec["candidate_type"] = detected["candidate_type"]
        spec["execution_mode"] = detected["execution_mode"]
        experiment["type"] = detected["experiment_type"]
        experiment["analysis_type"] = detected.get("analysis_type")

        training = spec.setdefault(
            "training",
            {},
        )

        training[
            "training_split"
        ] = "development_train"

        training[
            "validation_split"
        ] = "development_validation"

        evaluation = spec.setdefault(
            "evaluation",
            {},
        )

        evaluation.setdefault(
            "metrics",
            [
                "one_step_relative_error",
                "rollout_relative_error_at_horizons",
            ],
        )

        evaluation.setdefault(
            "rollout_horizons",
            [
                1,
                5,
                10,
                20,
                40,
                80,
            ],
        )

        evaluation.setdefault(
            "baselines",
            [
                "global_linear_operator",
                "persistence",
            ],
        )

        if experiment.get("type") == "diagnostic_analysis":
            if experiment.get("analysis_type") == "operator_decomposition":
                evaluation["metrics"] = [
                    "operator_frobenius_norm",
                    "symmetric_fraction",
                    "skew_fraction",
                    "bandwise_skew_fraction",
                    "bandwise_symmetric_fraction",
                ]
                falsification_hint = (
                    "Use the diagnostic outputs to determine whether the rotational/skew and "
                    "dissipative/symmetric components are negligible or structured across POD bands."
                )
            else:
                falsification_hint = (
                    "Use the diagnostic outputs to assess the proposed mechanism."
                )
        else:
            falsification_hint = (
                "Candidate improves validation prediction relative to the global linear baseline."
            )

        policy = spec.setdefault(
            "data_policy",
            {},
        )

        policy["allowed_split"] = [
            "development_train",
            "development_validation",
        ]

        policy["forbidden_split"] = [
            "final_test",
        ]

        policy["final_test_access"] = False

        falsification = spec.setdefault(
            "falsification",
            {},
        )

        falsification.setdefault(
            "success_condition",
            falsification_hint,
        )

        falsification.setdefault(
            "failure_condition",
            (
                "Diagnostic evidence indicates that the proposed decomposition is negligible "
                "or lacks reproducible band structure."
                if experiment.get("type") == "diagnostic_analysis"
                else "Candidate materially worsens one-step validation error or fails all rollout comparisons."
            ),
        )

        implementation_plan = spec.setdefault(
            "implementation_plan",
            {},
        )

        implementation_plan.setdefault(
            "modules",
            [],
        )

        implementation_plan.setdefault(
            "pseudocode",
            [],
        )

        self._infer_requirements(
            hypothesis,
            spec,
        )

        return spec

    # ========================================================
    # Prompt
    # ========================================================

    def generate_contract(
        self,
        hypothesis: dict[str, Any],
        state: dict[str, Any],
    ) -> str:

        capability = (
            load_capability_profile()
        )

        prompt = f"""
你是 Autonomous Scientific Discovery Harness 中的 Algorithm Agent。

你的任务：
把 Research Agent 提出的科学假设，转换成严格、可验证的 Algorithm Contract。

不要发明另一个算法。
不要修改科学假设的核心机制。
不要为了让模型可运行而偷偷替换机制。

当前实验能力：

{json.dumps(
    capability,
    ensure_ascii=False,
    indent=2,
)}

Research hypothesis：

{json.dumps(
    hypothesis,
    ensure_ascii=False,
    indent=2,
)}

Research state：

{json.dumps(
    state,
    ensure_ascii=False,
    indent=2,
)}

重要要求：

1. latent_dimension 必须使用当前能力 profile 中的实际值。
2. representation.normalization 使用简单 ASCII 单行文本。
3. 不要在 normalization 或科学描述中提及 final-test。
4. data_policy 必须固定：
   allowed_split = development_train, development_validation
   forbidden_split = final_test
   final_test_access = false
5. 如果需要历史状态，写入 implementation_requirements.required_history。
6. 如果需要外部参数，写入 required_external_parameters。
7. 如果需要速度分量、散度、压力等当前不存在的数据，写入 required_field_components。
8. 不要假装缺失的数据存在。
9. novelty_status 必须是 verification_incomplete。
10. 数学公式使用 ASCII，不使用 LaTeX，不使用反斜杠。
11. 只输出一个 JSON object。
12. 不输出 Markdown，不输出解释。

JSON schema：

{{
  "candidate_id": "",
  "name": "",
  "paradigm": "classical",
  "scientific_claim": "",

  "hypothesis": {{
    "source": "research_agent",
    "motivation": "",
    "novelty_status": "verification_incomplete"
  }},

  "representation": {{
    "input_state": "z_t",
    "latent_dimension": 52,
    "normalization": ""
  }},

  "dynamics": {{
    "equation": "z_next = A @ z_t",
    "inputs": [],
    "outputs": [],
    "parameters": [],
    "operator_structure": "",
    "stability_mechanism": ""
  }},

  "implementation_requirements": {{
    "required_inputs": ["latent_state"],
    "required_history": 0,
    "required_representation": "POD_latent",
    "required_latent_dimension": 52,
    "required_external_parameters": [],
    "required_field_components": []
  }},

  "training": {{
    "training_split": "development_train",
    "validation_split": "development_validation",
    "loss": "",
    "optimizer": "",
    "max_epochs": 100,
    "early_stopping": true
  }},

  "evaluation": {{
    "metrics": [],
    "rollout_horizons": [1,5,10,20,40,80],
    "baselines": ["global_linear_operator", "persistence"]
  }},

  "data_policy": {{
    "allowed_split": [
      "development_train",
      "development_validation"
    ],
    "forbidden_split": ["final_test"],
    "final_test_access": false
  }},

  "falsification": {{
    "success_condition": "",
    "failure_condition": ""
  }},

  "implementation_plan": {{
    "modules": [],
    "pseudocode": []
  }}
}}
""".strip()

        content = self.client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an Algorithm Contract Agent. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        if not content:
            raise RuntimeError(
                "MiniMax returned empty content."
            )

        return content

    # ========================================================
    # Run
    # ========================================================

    def run(
        self,
        research_dir: Path,
        output_dir: Path,
    ) -> dict[str, Any]:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print(
            "[AlgorithmAgent] "
            "Loading research state..."
        )

        try:
            hypotheses, state = (
                self.load_research(
                    research_dir
                )
            )
        except Exception as exc:
            result = {
                "status":
                    "research_load_failed",
                "error":
                    str(exc),
            }

            self.save_json(
                result,
                output_dir
                / "algorithm_result.json",
            )

            return result

        hypothesis = (
            self._load_active_hypothesis(
                research_dir,
                hypotheses,
            )
        )

        if not hypothesis:
            result = {
                "status":
                    "no_active_hypothesis",
                "error":
                    (
                        "ResearchAgent did not specify "
                        "a unique active hypothesis."
                    ),
            }

            self.save_json(
                result,
                output_dir
                / "algorithm_result.json",
            )

            return result

        hypothesis_source = (
            "research_agent"
        )

        self.save_json(
            {
                "hypothesis_source":
                    hypothesis_source,
                "hypothesis":
                    hypothesis,
            },
            output_dir
            / "hypothesis_used.json",
        )

        print()
        print(
            "[AlgorithmAgent] "
            "Using hypothesis from Research Agent."
        )

        print(
            "[AlgorithmAgent] Selected hypothesis:"
        )

        print(
            hypothesis.get(
                "name",
                "unknown",
            )
        )

        print(
            "[AlgorithmAgent] Source: research_agent"
        )

        print()
        print(
            "[AlgorithmAgent] "
            "Generating algorithm contract..."
        )

        try:
            raw = self.generate_contract(
                hypothesis=hypothesis,
                state=state,
            )

        except Exception as exc:
            result = {
                "status":
                    "algorithm_generation_failed",
                "error":
                    str(exc),
                "hypothesis":
                    hypothesis,
            }

            self.save_json(
                result,
                output_dir
                / "algorithm_generation_error.json",
            )

            self.save_json(
                result,
                output_dir
                / "algorithm_result.json",
            )

            return result

        raw_path = (
            research_dir
            / "candidates"
            / "algorithm_raw_response.txt"
        )

        raw_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw_path.write_text(
            raw,
            encoding="utf-8",
        )

        try:
            spec = self._parse_json(
                raw,
                allow_repair=True,
            )

        except Exception as exc:
            result = {
                "status":
                    "contract_parse_failed",
                "error":
                    str(exc),
                "hypothesis":
                    hypothesis,
                "raw_response":
                    str(raw_path),
            }

            self.save_json(
                result,
                output_dir
                / "algorithm_generation_error.json",
            )

            self.save_json(
                result,
                output_dir
                / "algorithm_result.json",
            )

            return result

        # Force contract identity + current representation.
        spec = self._normalize_spec(
            spec,
            hypothesis,
        )

        self.save_json(
            spec,
            output_dir
            / "algorithm_spec_raw.json",
        )

        print()
        print(
            "[AlgorithmAgent] "
            "Validating algorithm contract..."
        )

        contract_path = (
            output_dir
            / "algorithm_spec.json"
        )

        try:
            warnings = save_validated_contract(
                spec,
                contract_path,
            )

        except ContractValidationError as exc:
            print()
            print(
                "[AlgorithmAgent] "
                "Contract validation failed:"
            )
            print(
                f"  {exc}"
            )

            result = {
                "status":
                    "contract_invalid",
                "error":
                    str(exc),
                "hypothesis":
                    hypothesis,
                "raw_spec":
                    spec,
            }

            self.save_json(
                result,
                output_dir
                / "algorithm_validation_error.json",
            )

            self.save_json(
                result,
                output_dir
                / "algorithm_result.json",
            )

            return result

        implementability = (
            spec.get(
                "implementability",
                {},
            )
        )

        # Do not send scientifically incompatible candidates to CodeAgent.
        if not implementability.get(
            "current_pipeline_compatible",
            True,
        ):

            missing = implementability.get(
                "missing_inputs",
                [],
            )

            print()
            print(
                "[AlgorithmAgent] "
                "Candidate is scientifically expressible "
                "but not executable under the current pipeline."
            )

            for item in missing:
                print(
                    f"  - {item}"
                )

            result = {
                "status":
                    "not_implementable",

                "candidate_id":
                    spec.get(
                        "candidate_id"
                    ),

                "hypothesis_id":
                    spec.get(
                        "hypothesis_id"
                    ),

                "name":
                    spec.get(
                        "name"
                    ),

                "hypothesis_source":
                    hypothesis_source,

                "hypothesis_file":
                    str(
                        output_dir
                        / "hypothesis_used.json"
                    ),

                "contract_path":
                    str(
                        contract_path
                    ),

                "warnings":
                    warnings,

                "missing_inputs":
                    missing,

                "implementability":
                    implementability,

                "data_policy":
                    spec.get(
                        "data_policy",
                        {},
                    ),
            }

            self.save_json(
                result,
                output_dir
                / "algorithm_result.json",
            )

            return result

        result = {
            "status":
                "success",

            "candidate_id":
                spec[
                    "candidate_id"
                ],

            "hypothesis_id":
                spec.get("hypothesis_id"),

            "name":
                spec["name"],

            "hypothesis_source":
                hypothesis_source,

            "hypothesis_file":
                str(
                    output_dir
                    / "hypothesis_used.json"
                ),

            "contract_path":
                str(
                    contract_path
                ),

            "warnings":
                warnings,

            "implementability":
                implementability,

            "data_policy":
                spec[
                    "data_policy"
                ],
        }

        self.save_json(
            result,
            output_dir
            / "algorithm_result.json",
        )

        print()
        print(
            "[AlgorithmAgent] "
            "Contract accepted."
        )

        print(
            f"[AlgorithmAgent] "
            f"Candidate: {spec['name']}"
        )

        if warnings:
            print()
            print(
                "[AlgorithmAgent] "
                "Warnings:"
            )

            for warning in warnings:
                print(
                    f"  - {warning}"
                )

        return result
