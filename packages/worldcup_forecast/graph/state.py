from __future__ import annotations

from operator import add
from typing import Annotated, Any

from typing_extensions import TypedDict


class ForecastState(TypedDict, total=False):
    # ── inputs ────────────────────────────────────────────────────────────────
    request: dict                       # MatchPredictionRequest.model_dump()
    llm_settings: dict                  # LLMSettings.model_dump()
    search_settings: dict               # SearchSettings.model_dump()

    # ── deterministic stats (written by supervisor_node) ──────────────────────
    probabilities: dict                 # {home_win, draw, away_win}
    expected_score: tuple[float, float]
    most_likely_score: str
    bet_signals: list[dict]
    odds: dict | None
    shap_values: dict[str, float]

    # ── per-agent findings (fan-in via Annotated[list, add] reducer) ──────────
    agent_findings: Annotated[list[dict], add]

    # ── debate intermediate (written by debate_node) ──────────────────────────
    bull_argument: str
    bear_argument: str

    # ── final output ──────────────────────────────────────────────────────────
    explanation: str
    report_id: str
