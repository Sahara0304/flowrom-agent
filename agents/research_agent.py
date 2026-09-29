"""Autonomous scientific hypothesis discovery for FlowROM-Agent."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from research.research_state import DISCOVERY_DIR, build_scientific_context, load_state, save_hypothesis_set, save_state
from tools.minimax_client import MiniMaxClient


class ResearchAgent:
    def __init__(self, project_root: Optional[Path] = None, minimax_api_key: Optional[str] = None,
                 model: str = "MiniMax-M3", base_url: Optional[str] = None,
                 semantic_scholar_key: Optional[str] = None):
        self.root = Path(project_root or Path(__file__).resolve().parents[1])
        self.model = model
        self.minimax_api_key = minimax_api_key or os.getenv("MINIMAX_API_KEY")
        if not self.minimax_api_key:
            raise RuntimeError("MINIMAX_API_KEY is not set")
        self.semantic_scholar_key = semantic_scholar_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.session = requests.Session()
        self.minimax = MiniMaxClient(
            api_key=self.minimax_api_key,
            base_url=(base_url or os.getenv("MINIMAX_BASE_URL")),
            model=self.model,
            connect_timeout=15.0,
            read_timeout=180.0,
            max_retries=3,
        )

    def _call_minimax(self, prompt: str, temperature: float = 0.3) -> str:
        return self.minimax.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an autonomous scientific discovery agent. "
                        "Return valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )

    def _semantic_scholar_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        headers = {}
        if self.semantic_scholar_key:
            headers["x-api-key"] = self.semantic_scholar_key
        params = {"query": query, "limit": limit, "fields": "title,abstract,year,authors,url"}
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=30)
                if r.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json().get("data", [])
            except requests.RequestException:
                if attempt == 2:
                    return []
                time.sleep(1.5 * (attempt + 1))
        return []

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        raise ValueError("MiniMax returned invalid JSON")

    @staticmethod
    def _normalize(payload: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        raw = payload.get("hypotheses", [])
        if not isinstance(raw, list):
            return []
        out = []
        for i, item in enumerate(raw[:limit]):
            if not isinstance(item, dict):
                continue
            h = dict(item)
            h.setdefault("id", f"H_auto_{i + 1}")
            h.setdefault("name", f"Autonomous hypothesis {i + 1}")
            h.setdefault("scientific_question", "What mechanism explains the observed prediction error?")
            h.setdefault("mechanism", "unspecified")
            h.setdefault("mathematical_form", "unspecified")
            h.setdefault("motivation", "Generated from current evidence.")
            h.setdefault("explanation_of_previous_failure", "Not specified.")
            h.setdefault("distinguishing_prediction", "Not specified.")
            h.setdefault("minimum_experiment", [])
            h.setdefault("expected_observation_if_true", "Not specified.")
            h.setdefault("expected_observation_if_false", "Not specified.")
            h.setdefault("risks", [])
            h.setdefault("relation_to_existing_methods", "Requires literature verification.")
            h.setdefault("novelty_status", "verification_incomplete")
            out.append(h)
        return out

    @staticmethod
    def _emergency_fallback(limit: int = 3) -> List[Dict[str, Any]]:
        # Only used when the LLM is unavailable. These are explicitly marked as fallback seeds.
        seeds = [
            {
                "id": "H_fallback_residual",
                "name": "Weak structured residual around global latent dynamics",
                "scientific_question": "Can a constrained residual capture repeatable deviations left by the global operator?",
                "mechanism": "Shared global operator plus low-complexity state-conditioned residual.",
                "mathematical_form": "z_next = A0 z_t + DeltaA(z_t) z_t",
                "motivation": "The global linear baseline is much stronger than the fully regime-conditioned model.",
                "explanation_of_previous_failure": "Independent local operators may have too much variance.",
                "distinguishing_prediction": "A constrained residual should improve validation without materially hurting one-step accuracy.",
                "minimum_experiment": ["Fit A0 on development_train", "fit residual on development_train", "evaluate on development_validation"],
                "expected_observation_if_true": "Small reproducible validation gain.",
                "expected_observation_if_false": "No gain or degradation.",
                "risks": ["overfitting"],
                "relation_to_existing_methods": "Related to residual operator learning; novelty requires verification.",
                "novelty_status": "verification_incomplete",
                "generation_mode": "emergency_fallback",
            },
            {
                "id": "H_fallback_memory",
                "name": "Higher-order latent memory",
                "scientific_question": "Is the current latent state insufficient to summarize recent history?",
                "mechanism": "Second-order latent autoregression.",
                "mathematical_form": "z_next = A0 z_t + A1 z_{t-1}",
                "motivation": "Good one-step global accuracy with accumulating rollout error leaves memory as a competing explanation.",
                "explanation_of_previous_failure": "A first-order Markov latent model may be missing temporal information.",
                "distinguishing_prediction": "Adding one lag should reduce multi-step error if memory matters.",
                "minimum_experiment": ["Fit A0,A1 on development_train", "evaluate one-step and rollout on development_validation"],
                "expected_observation_if_true": "Lower long-horizon validation error.",
                "expected_observation_if_false": "No improvement over global linear.",
                "risks": ["parameter growth", "instability"],
                "relation_to_existing_methods": "Related to autoregressive ROMs; novelty requires verification.",
                "novelty_status": "verification_incomplete",
                "generation_mode": "emergency_fallback",
            },
            {
                "id": "H_fallback_lift",
                "name": "Low-dimensional nonlinear lifting",
                "scientific_question": "Does a compact nonlinear feature space linearize the latent evolution better?",
                "mechanism": "Feature lifting followed by linear dynamics.",
                "mathematical_form": "phi(z_next) approximately equals K phi(z_t)",
                "motivation": "The global linear law may capture the dominant component while leaving structured nonlinear residuals.",
                "explanation_of_previous_failure": "Hard regime switching is only one possible nonlinear mechanism.",
                "distinguishing_prediction": "A compact lift should improve rollout at modest complexity.",
                "minimum_experiment": ["construct train-only feature lift", "fit K on development_train", "evaluate on development_validation"],
                "expected_observation_if_true": "Improved rollout accuracy.",
                "expected_observation_if_false": "No gain or poorer generalization.",
                "risks": ["feature explosion", "overfitting"],
                "relation_to_existing_methods": "Related to Koopman/lifting methods; novelty requires verification.",
                "novelty_status": "verification_incomplete",
                "generation_mode": "emergency_fallback",
            },
        ]
        return seeds[:limit]

    def discover_next(self, diagnostics: Optional[Dict[str, Any]] = None, max_hypotheses: int = 3) -> Dict[str, Any]:
        state = load_state()
        context = build_scientific_context(diagnostics=diagnostics)

        literature = []
        for query in (
            "latent linear dynamics reduced order modeling memory fluid flow",
            "nonlinear latent dynamics operator learning reduced order model fluid",
        ):
            literature.extend(self._semantic_scholar_search(query, limit=5))
            time.sleep(1.2)

        payload = {
            "context": context,
            "literature": literature,
            "instruction": "Generate genuinely different, falsifiable next hypotheses. Do not perform hyperparameter search disguised as discovery.",
        }
        prompt = f"""
You are the Research Agent in an autonomous scientific discovery system for ROM.

Previous hypotheses and experiments are evidence. A failed mechanism must not simply be retried with a different
number of clusters or regularization. Generate 3-{max_hypotheses} genuinely different scientific hypotheses.
Each hypothesis must introduce a changed mechanism, representation, memory structure, operator decomposition,
conditioning rule, or mathematical formulation.

Return JSON only:
{{
  "research_strategy": {{"mode": "parallel_exploration", "reason": "..."}},
  "unresolved_questions": ["..."],
  "hypotheses": [
    {{
      "id": "H2",
      "name": "...",
      "scientific_question": "...",
      "mechanism": "...",
      "mathematical_form": "...",
      "motivation": "...",
      "explanation_of_previous_failure": "...",
      "distinguishing_prediction": "...",
      "minimum_experiment": ["..."],
      "expected_observation_if_true": "...",
      "expected_observation_if_false": "...",
      "risks": ["..."],
      "relation_to_existing_methods": "...",
      "novelty_status": "verification_incomplete"
    }}
  ]
}}

Constraints:
- Use development_train/development_validation only.
- Never read final-test data.
- Never claim proven novelty.
- Explicitly explain why each hypothesis is different from already falsified mechanisms.
- Prefer experiments that discriminate between competing explanations.

Evidence:
{json.dumps(payload, ensure_ascii=False, indent=2)[:60000]}
"""

        try:
            raw = self._call_minimax(prompt)
            obj = self._parse_json(raw)
            hypotheses = self._normalize(obj, max_hypotheses)
            strategy = obj.get("research_strategy", {})
            unresolved = obj.get("unresolved_questions", [])
            source = "llm"
        except Exception as exc:
            hypotheses = self._emergency_fallback(max_hypotheses)
            strategy = {"mode": "emergency_fallback", "reason": str(exc), "llm_available": False}
            unresolved = [
                "Does the global linear model miss temporal memory?",
                "Is there structured nonlinear residual dynamics after the global component?",
            ]
            source = "fallback"
            (DISCOVERY_DIR / "last_research_error.txt").write_text(str(exc), encoding="utf-8")

        cycle = int(state.get("research_cycle", 0)) + 1
        artifact = save_hypothesis_set(cycle, hypotheses, {**strategy, "source": source, "literature_count": len(literature)})
        state["research_cycle"] = cycle
        state["status"] = "hypotheses_generated"
        state["active_hypotheses"] = [h.get("id") for h in hypotheses]
        state["unresolved_questions"] = unresolved
        save_state(state)
        return {
            "research_cycle": cycle,
            "strategy": strategy,
            "hypotheses": hypotheses,
            "artifact": str(artifact),
            "literature_count": len(literature),
            "generation_source": source,
        }


if __name__ == "__main__":
    print(json.dumps(ResearchAgent().discover_next(), ensure_ascii=False, indent=2))
