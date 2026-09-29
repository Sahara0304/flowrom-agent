"""Persistent scientific memory for FlowROM-Agent.

The state layer is deliberately independent from the LLM. It records:
- research cycles
- candidate hypotheses
- experiment verdicts
- unresolved questions
- artifact locations

It never reads final-test data.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "research"
STATE_PATH = RESEARCH_DIR / "research_state.json"
HYPOTHESES_PATH = RESEARCH_DIR / "research_hypotheses.json"
ACTIVE_HYPOTHESIS_PATH = RESEARCH_DIR / "active_hypothesis.json"
HISTORY_DIR = RESEARCH_DIR / "history"
DISCOVERY_DIR = RESEARCH_DIR / "discovery_cycles"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for path in (RESEARCH_DIR, HISTORY_DIR, DISCOVERY_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    tmp.replace(path)


def default_state() -> Dict[str, Any]:
    timestamp = now_iso()

    return {
        "version": 3,
        "research_cycle": 0,
        "status": "initialized",

        "current_question": None,
        "current_hypothesis": None,

        "active_hypotheses": [],
        "hypothesis_history": [],

        "unresolved_questions": [],
        "research_strategy": None,

        "created_at": timestamp,
        "updated_at": timestamp,
    }


def load_state() -> Dict[str, Any]:
    ensure_dirs()

    state = load_json(
        STATE_PATH,
        default_state(),
    )

    if not isinstance(state, dict):
        state = default_state()

    defaults = default_state()

    for key, value in defaults.items():
        state.setdefault(key, value)

    state["updated_at"] = now_iso()

    return state


def save_state(state: Dict[str, Any]) -> None:
    state = dict(state)

    state["updated_at"] = now_iso()

    save_json(
        STATE_PATH,
        state,
    )


def save_active_hypothesis(hypothesis: Dict[str, Any]) -> None:
    save_json(
        ACTIVE_HYPOTHESIS_PATH,
        hypothesis,
    )


def set_active_hypotheses(
    hypotheses: List[Dict[str, Any]],
    strategy: Optional[Dict[str, Any]] = None,
) -> None:
    state = load_state()

    ids = [
        str(h.get("id"))
        for h in hypotheses
        if h.get("id")
    ]

    state["active_hypotheses"] = ids

    state["research_strategy"] = (
        strategy
        if strategy is not None
        else state.get("research_strategy")
    )

    state["status"] = "hypotheses_generated"

    save_state(state)


def activate_hypothesis_for_execution(
    hypothesis: Dict[str, Any],
) -> None:
    """
    Set the single currently executing hypothesis without
    losing the remaining candidates of the current cycle.
    """

    state = load_state()

    state["current_hypothesis"] = hypothesis
    state["status"] = "testing_hypothesis"

    save_state(state)

    save_active_hypothesis(hypothesis)


def restore_cycle_status() -> None:
    state = load_state()

    state["current_hypothesis"] = None

    remaining = list(
        state.get("active_hypotheses", [])
    )

    if remaining:
        state["status"] = "testing_cycle"
    else:
        state["status"] = "cycle_complete"

    save_state(state)

    try:
        ACTIVE_HYPOTHESIS_PATH.unlink()
    except FileNotFoundError:
        pass


def save_hypothesis_set(
    cycle: int,
    hypotheses: List[Dict[str, Any]],
    strategy: Optional[Dict[str, Any]] = None,
    literature: Optional[List[Dict[str, Any]]] = None,
) -> Path:

    payload = {
        "research_cycle": cycle,
        "generated_at": now_iso(),
        "strategy": strategy or {},
        "hypotheses": hypotheses,
        "literature": literature or [],
    }

    save_json(
        HYPOTHESES_PATH,
        payload,
    )

    out = (
        DISCOVERY_DIR
        / f"cycle_{cycle:03d}.json"
    )

    save_json(
        out,
        payload,
    )

    return out


def record_hypothesis_outcome(
    hypothesis: Dict[str, Any],
    verdict: Dict[str, Any],
    artifact_dir: Optional[str] = None,
) -> None:

    state = load_state()

    hypothesis_id = hypothesis.get("id")

    entry = {
        "hypothesis_id": hypothesis_id,
        "name": hypothesis.get("name"),

        "status": verdict.get(
            "status",
            "unknown",
        ),

        "scientific_claim_status": verdict.get(
            "scientific_claim_status"
        ),

        "verdict": verdict,

        "artifact_dir": artifact_dir,

        "recorded_at": now_iso(),
    }

    # Remove an earlier record for the same hypothesis
    # if this is a retry.
    history = [
        item
        for item in state.get(
            "hypothesis_history",
            [],
        )
        if item.get(
            "hypothesis_id"
        ) != hypothesis_id
    ]

    history.append(entry)

    state["hypothesis_history"] = history

    # Current candidate is no longer active.
    state["active_hypotheses"] = [
        item
        for item in state.get(
            "active_hypotheses",
            [],
        )
        if item != hypothesis_id
    ]

    unresolved = verdict.get(
        "unresolved_questions"
    )

    if isinstance(unresolved, list):
        state["unresolved_questions"] = unresolved

    state["current_hypothesis"] = None

    if state["active_hypotheses"]:
        state["status"] = "testing_cycle"
    else:
        state["status"] = "cycle_complete"

    save_state(state)

    try:
        ACTIVE_HYPOTHESIS_PATH.unlink()
    except FileNotFoundError:
        pass


def history_records() -> List[Dict[str, Any]]:
    return load_state().get(
        "hypothesis_history",
        [],
    )


def failed_records() -> List[Dict[str, Any]]:
    return [
        item
        for item in history_records()
        if str(
            item.get("status", "")
        ).lower()
        in {
            "falsified",
            "execution_failed",
        }
    ]


def supported_records() -> List[Dict[str, Any]]:
    return [
        item
        for item in history_records()
        if str(
            item.get("status", "")
        ).lower()
        == "supported"
    ]


def next_cycle_number() -> int:
    return int(
        load_state().get(
            "research_cycle",
            0,
        )
    ) + 1


def build_scientific_context(
    diagnostics: Optional[Dict[str, Any]] = None,
    recent_experiments: Optional[
        List[Dict[str, Any]]
    ] = None,
) -> Dict[str, Any]:

    state = load_state()

    history = history_records()

    return {
        "research_state": state,

        "diagnostics": diagnostics or {},

        "recent_experiments": (
            recent_experiments
            or history[-12:]
        ),

        "failed_hypotheses": (
            failed_records()[-12:]
        ),

        "supported_hypotheses": (
            supported_records()[-12:]
        ),

        "unresolved_questions": state.get(
            "unresolved_questions",
            [],
        ),
    }