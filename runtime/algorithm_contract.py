from __future__ import annotations

"""Algorithm Contract validation for FlowROM autonomous discovery.

This module is intentionally structural.  It validates the machine-readable
contract fields that control execution and data access, rather than scanning
all narrative strings for words such as ``final_test``.  That distinction is
important because scientific descriptions may legitimately mention the final
test split while still obeying the data policy.
"""

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATH = ROOT / "runtime" / "capability_profile.json"


class ContractValidationError(ValueError):
    """Raised when an Algorithm Contract violates the execution contract."""


# Backward-compatible exception name used by older scripts/agents.
# This is an alias, not a subclass, so both names catch the same exceptions.
AlgorithmContractError = ContractValidationError


def _load_capability() -> dict[str, Any]:
    if not CAPABILITY_PATH.exists():
        return {
            "representation": {"type": "POD_latent", "latent_dimension": 52},
            "data_policy": {
                "allowed_splits": [
                    "development_train",
                    "development_validation",
                ],
                "forbidden_splits": ["final_test"],
                "final_test_access": False,
            },
        }

    try:
        with CAPABILITY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ContractValidationError(
            f"无法读取 capability_profile.json: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ContractValidationError("capability_profile.json 必须是 JSON object。")
    return data


def _require_dict(obj: Any, name: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ContractValidationError(f"{name} 必须是 JSON object。")
    return obj


def _require_nonempty_string(obj: dict[str, Any], key: str, path: str) -> None:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{path}.{key} 必须是非空字符串。")


def _as_positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise ContractValidationError(f"{path} 必须是正整数。")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{path} 必须是正整数。") from exc
    if integer <= 0:
        raise ContractValidationError(f"{path} 必须是正整数。")
    return integer


def _normalize_unique_strings(value: Any, path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ContractValidationError(f"{path} 必须是字符串数组。")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ContractValidationError(
                f"{path} 中的元素必须是非空字符串。"
            )
        result.append(item.strip())
    return list(dict.fromkeys(result))


def validate_algorithm_contract(spec: dict[str, Any]) -> list[str]:
    """Validate an Algorithm Contract and return non-fatal warnings."""

    if not isinstance(spec, dict):
        raise ContractValidationError("Algorithm Contract 必须是 JSON object。")

    warnings: list[str] = []

    # ------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------
    _require_nonempty_string(spec, "candidate_id", "contract")
    _require_nonempty_string(spec, "name", "contract")

    # ------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------
    representation = _require_dict(
        spec.get("representation", {}),
        "representation",
    )

    rep_type = representation.get("type", "POD_latent")
    if not isinstance(rep_type, str) or not rep_type.strip():
        raise ContractValidationError(
            "representation.type 必须是非空字符串。"
        )

    latent_dimension = _as_positive_int(
        representation.get("latent_dimension"),
        "representation.latent_dimension",
    )

    capability = _load_capability()
    cap_rep = capability.get("representation", {})
    cap_dim = cap_rep.get("latent_dimension")
    if cap_dim is not None:
        try:
            cap_dim = int(cap_dim)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(
                "capability_profile.representation.latent_dimension 无效。"
            ) from exc
        if latent_dimension != cap_dim:
            raise ContractValidationError(
                "representation.latent_dimension "
                f"({latent_dimension}) 与当前 capability profile "
                f"({cap_dim}) 不一致。"
            )

    normalization = representation.get("normalization", "")
    if not isinstance(normalization, str):
        raise ContractValidationError(
            "representation.normalization 必须是字符串。"
        )
    if "\n" in normalization or "\r" in normalization:
        raise ContractValidationError(
            "representation.normalization 必须是单行文本。"
        )

    # IMPORTANT: do NOT scan the narrative text for forbidden split names.
    # Data isolation is controlled structurally below.

    # ------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------
    dynamics = _require_dict(spec.get("dynamics", {}), "dynamics")
    equation = dynamics.get("equation", "")
    if not isinstance(equation, str):
        raise ContractValidationError("dynamics.equation 必须是字符串。")
    dynamics["inputs"] = _normalize_unique_strings(
        dynamics.get("inputs", []),
        "dynamics.inputs",
    )
    dynamics["outputs"] = _normalize_unique_strings(
        dynamics.get("outputs", []),
        "dynamics.outputs",
    )
    dynamics["parameters"] = _normalize_unique_strings(
        dynamics.get("parameters", []),
        "dynamics.parameters",
    )

    # ------------------------------------------------------------
    # Implementation requirements
    # ------------------------------------------------------------
    requirements = spec.get("implementation_requirements", {})
    if requirements is None:
        requirements = {}
        spec["implementation_requirements"] = requirements
    requirements = _require_dict(
        requirements,
        "implementation_requirements",
    )

    requirements["required_inputs"] = _normalize_unique_strings(
        requirements.get("required_inputs", ["latent_state"]),
        "implementation_requirements.required_inputs",
    )

    required_history = requirements.get("required_history", 0)
    if isinstance(required_history, bool):
        raise ContractValidationError(
            "implementation_requirements.required_history 必须是非负整数。"
        )
    try:
        required_history = int(required_history)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            "implementation_requirements.required_history 必须是非负整数。"
        ) from exc
    if required_history < 0:
        raise ContractValidationError(
            "implementation_requirements.required_history 必须是非负整数。"
        )
    requirements["required_history"] = required_history

    required_rep = requirements.get("required_representation", rep_type)
    if not isinstance(required_rep, str) or not required_rep.strip():
        raise ContractValidationError(
            "implementation_requirements.required_representation 必须是字符串。"
        )
    if required_rep != rep_type:
        raise ContractValidationError(
            "implementation_requirements.required_representation "
            f"({required_rep}) 与 representation.type ({rep_type}) 不一致。"
        )

    required_dim = _as_positive_int(
        requirements.get("required_latent_dimension", latent_dimension),
        "implementation_requirements.required_latent_dimension",
    )
    if required_dim != latent_dimension:
        raise ContractValidationError(
            "implementation_requirements.required_latent_dimension "
            "必须与 representation.latent_dimension 一致。"
        )
    requirements["required_latent_dimension"] = required_dim

    requirements["required_external_parameters"] = _normalize_unique_strings(
        requirements.get("required_external_parameters", []),
        "implementation_requirements.required_external_parameters",
    )
    requirements["required_field_components"] = _normalize_unique_strings(
        requirements.get("required_field_components", []),
        "implementation_requirements.required_field_components",
    )

    # ------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------
    training = _require_dict(spec.get("training", {}), "training")
    train_split = training.get("training_split", "development_train")
    val_split = training.get("validation_split", "development_validation")
    if train_split != "development_train":
        raise ContractValidationError(
            "training.training_split 必须是 development_train。"
        )
    if val_split != "development_validation":
        raise ContractValidationError(
            "training.validation_split 必须是 development_validation。"
        )

    evaluation = _require_dict(spec.get("evaluation", {}), "evaluation")
    evaluation["metrics"] = _normalize_unique_strings(
        evaluation.get("metrics", []),
        "evaluation.metrics",
    )
    horizons = evaluation.get("rollout_horizons", [1, 5, 10, 20, 40, 80])
    if not isinstance(horizons, list) or not horizons:
        raise ContractValidationError(
            "evaluation.rollout_horizons 必须是非空数组。"
        )
    for h in horizons:
        if isinstance(h, bool):
            raise ContractValidationError(
                "evaluation.rollout_horizons 必须包含正整数。"
            )
        try:
            ih = int(h)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(
                "evaluation.rollout_horizons 必须包含正整数。"
            ) from exc
        if ih <= 0:
            raise ContractValidationError(
                "evaluation.rollout_horizons 必须包含正整数。"
            )
    evaluation["baselines"] = _normalize_unique_strings(
        evaluation.get("baselines", []),
        "evaluation.baselines",
    )

    # ------------------------------------------------------------
    # Data policy: structural enforcement only
    # ------------------------------------------------------------
    policy = _require_dict(spec.get("data_policy", {}), "data_policy")
    allowed = policy.get(
        "allowed_split",
        ["development_train", "development_validation"],
    )
    forbidden = policy.get("forbidden_split", ["final_test"])
    access = policy.get("final_test_access", False)

    if allowed != ["development_train", "development_validation"]:
        raise ContractValidationError(
            "data_policy.allowed_split 必须且只能为 "
            "['development_train', 'development_validation']。"
        )
    if forbidden != ["final_test"]:
        raise ContractValidationError(
            "data_policy.forbidden_split 必须为 ['final_test']。"
        )
    if access is not False:
        raise ContractValidationError(
            "data_policy.final_test_access 必须为 false。"
        )

    # Optional top-level execution mode.  This is future-compatible with
    # diagnostic experiments, but the existing model runner remains valid.
    candidate_type = spec.get("candidate_type")
    if candidate_type is not None and not isinstance(candidate_type, str):
        raise ContractValidationError("candidate_type 必须是字符串。")

    execution_mode = spec.get("execution_mode")
    if execution_mode is not None and not isinstance(execution_mode, str):
        raise ContractValidationError("execution_mode 必须是字符串。")

    # ------------------------------------------------------------
    # Implementability consistency checks
    # ------------------------------------------------------------
    impl = spec.get("implementability", {})
    if impl is not None:
        impl = _require_dict(impl, "implementability")
        missing = impl.get("missing_inputs", [])
        impl["missing_inputs"] = _normalize_unique_strings(
            missing,
            "implementability.missing_inputs",
        )
        if "current_pipeline_compatible" in impl and not isinstance(
            impl["current_pipeline_compatible"], bool
        ):
            raise ContractValidationError(
                "implementability.current_pipeline_compatible 必须是布尔值。"
            )

    # ------------------------------------------------------------
    # Falsification / implementation plan are descriptive, so validate type
    # but do not impose scientific semantics.
    # ------------------------------------------------------------
    for section_name in ("hypothesis", "falsification", "implementation_plan"):
        if section_name in spec and spec[section_name] is not None:
            _require_dict(spec[section_name], section_name)

    # A warning is preferable to a hard failure when an implementation asks
    # for unavailable capabilities. AlgorithmAgent already marks such specs as
    # not_implementable instead of silently changing the hypothesis.
    available = set(capability.get("available_inputs", []))
    missing_capabilities: list[str] = []
    for item in requirements["required_inputs"]:
        if item not in available:
            missing_capabilities.append(item)
    for item in requirements["required_external_parameters"]:
        if item not in available:
            missing_capabilities.append(item)
    for item in requirements["required_field_components"]:
        if item not in available:
            missing_capabilities.append(item)

    runner = capability.get("runner", {})
    if required_history > 0 and not runner.get("explicit_history_input", False):
        missing_capabilities.append("runner_explicit_history_input")

    if missing_capabilities:
        warnings.append(
            "Candidate requires capabilities unavailable in the current runner: "
            + ", ".join(sorted(set(missing_capabilities)))
        )

    return warnings


def load_capability_profile() -> dict[str, Any]:
    """Public compatibility wrapper for capability loading."""
    return _load_capability()


def detect_final_test_access(spec: dict[str, Any]) -> list[str]:
    """Compatibility API using structural policy checks only.

    Narrative mentions of ``final_test`` are intentionally ignored.
    """
    violations: list[str] = []
    policy = spec.get("data_policy", {}) if isinstance(spec, dict) else {}
    if not isinstance(policy, dict):
        return ["data_policy"]
    if policy.get("final_test_access") is not False:
        violations.append("data_policy.final_test_access")
    if policy.get("forbidden_split") != ["final_test"]:
        violations.append("data_policy.forbidden_split")
    if policy.get("allowed_split") != ["development_train", "development_validation"]:
        violations.append("data_policy.allowed_split")
    return violations


def validate_data_policy(spec: dict[str, Any]) -> None:
    """Compatibility wrapper for structural data-policy validation."""
    policy = spec.get("data_policy", {}) if isinstance(spec, dict) else {}
    if not isinstance(policy, dict):
        raise ContractValidationError("data_policy 必须是 JSON object。")
    allowed = policy.get("allowed_split", ["development_train", "development_validation"])
    forbidden = policy.get("forbidden_split", ["final_test"])
    access = policy.get("final_test_access", False)
    if allowed != ["development_train", "development_validation"]:
        raise ContractValidationError(
            "data_policy.allowed_split 必须且只能为 ['development_train', 'development_validation']。"
        )
    if forbidden != ["final_test"]:
        raise ContractValidationError(
            "data_policy.forbidden_split 必须为 ['final_test']。"
        )
    if access is not False:
        raise ContractValidationError(
            "data_policy.final_test_access 必须为 false。"
        )


def validate_implementability(spec: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper exposing capability checks."""
    if not isinstance(spec, dict):
        raise ContractValidationError("Algorithm Contract 必须是 JSON object。")
    capability = _load_capability()
    requirements = spec.get("implementation_requirements", {}) or {}
    if not isinstance(requirements, dict):
        raise ContractValidationError("implementation_requirements 必须是 JSON object。")

    missing: list[str] = []
    available = set(capability.get("available_inputs", []))
    unavailable = set(capability.get("unavailable_inputs", []))

    for item in requirements.get("required_inputs", []) or []:
        if item not in available:
            missing.append(str(item))
    for item in requirements.get("required_external_parameters", []) or []:
        if item not in available or item in unavailable:
            missing.append(str(item))
    for item in requirements.get("required_field_components", []) or []:
        if item not in available or item in unavailable:
            missing.append(str(item))

    required_history = int(requirements.get("required_history", 0) or 0)
    runner = capability.get("runner", {}) or {}
    if required_history > 0 and not runner.get("explicit_history_input", False):
        missing.append("runner_explicit_history_input")

    rep = capability.get("representation", {}) or {}
    required_dim = requirements.get("required_latent_dimension")
    if required_dim is not None and rep.get("latent_dimension") is not None:
        if int(required_dim) != int(rep.get("latent_dimension")):
            missing.append(
                f"latent_dimension:{required_dim}!=current:{rep.get('latent_dimension')}"
            )
    required_rep = requirements.get("required_representation")
    if required_rep is not None and required_rep != rep.get("type"):
        missing.append(f"representation:{required_rep}")

    result = spec.setdefault("implementability", {})
    if not isinstance(result, dict):
        raise ContractValidationError("implementability 必须是 JSON object。")
    result.update({
        "current_pipeline_compatible": len(missing) == 0,
        "missing_inputs": sorted(set(missing)),
        "capability_profile": str(CAPABILITY_PATH),
        "required_history": required_history,
    })
    return result


def save_validated_contract(
    spec: dict[str, Any],
    path: str | Path,
) -> list[str]:
    """Validate then atomically write an Algorithm Contract JSON file."""

    warnings = validate_algorithm_contract(spec)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            spec,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    tmp.replace(target)
    return warnings
