from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class LLMSettings(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4.1-mini"
    temperature: float = Field(default=0.2, ge=0, le=2)
    timeout_seconds: int = Field(default=30, ge=1, le=180)
    enabled: bool = False


class PublicLLMSettings(BaseModel):
    base_url: str
    api_key_masked: str
    api_key_saved: bool = False
    model: str
    temperature: float
    timeout_seconds: int
    enabled: bool


class SearchSettings(BaseModel):
    """Configuration for the pluggable web-search layer (decoupled from the LLM).

    ``provider`` selects the request/response adapter. ``bocha`` and ``zhipu``
    are JSON search APIs reachable from mainland China; ``none`` disables search.
    """

    provider: Literal["none", "bocha", "zhipu", "custom"] = "none"
    base_url: str = "https://api.bochaai.com/v1/web-search"
    api_key: str = ""
    timeout_seconds: int = Field(default=15, ge=1, le=60)
    max_results: int = Field(default=6, ge=1, le=20)
    enabled: bool = False


class PublicSearchSettings(BaseModel):
    provider: str
    base_url: str
    api_key_masked: str
    api_key_saved: bool = False
    timeout_seconds: int
    max_results: int
    enabled: bool


class OddsRecord(BaseModel):
    match_id: str
    kickoff_time: datetime | None = None
    home_team: str
    away_team: str
    play_type: str
    handicap: str | None = None
    win_odds: float | None = None
    draw_odds: float | None = None
    lose_odds: float | None = None
    source: str = "500.com"
    source_url: str
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    raw: dict[str, Any] = Field(default_factory=dict)


class AgentFinding(BaseModel):
    agent: str
    confidence: float = Field(ge=0, le=1)
    signal: Literal["positive", "neutral", "negative"]
    rationale: str
    sources: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class ReasoningStep(BaseModel):
    """One step in an agent's exposed chain of reasoning."""

    kind: Literal["observation", "analysis", "conclusion"]
    content: str


class AgentReasoning(BaseModel):
    """An agent's finding enriched with an explicit, inspectable reasoning trace."""

    agent: str
    confidence: float = Field(ge=0, le=1)
    signal: Literal["positive", "neutral", "negative"]
    steps: list[ReasoningStep] = Field(default_factory=list)
    rationale: str = ""
    sources: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    powered_by: Literal["llm", "deterministic"] = "deterministic"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    bankroll: float = Field(default=10000, gt=0)


class AskResponse(BaseModel):
    question: str
    home_team: str | None = None
    away_team: str | None = None
    answer: str
    matched: bool = False


class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    kickoff_time: datetime | None = None
    neutral_site: bool = True
    bankroll: float = Field(default=10000, gt=0)
    kelly_fraction: float = Field(default=0.25, ge=0, le=1)
    max_stake_fraction: float = Field(default=0.05, ge=0, le=0.5)
    value_edge_threshold: float = Field(default=0.025, ge=-0.2, le=0.5)
    home_advantage_elo: float = Field(default=45, ge=0, le=200)
    draw_bias: float = Field(default=1.0, ge=0.5, le=1.8)
    goal_rate_multiplier: float = Field(default=1.0, ge=0.5, le=1.8)


class OutcomeProbability(BaseModel):
    home_win: float
    draw: float
    away_win: float


class BetSignal(BaseModel):
    outcome: Literal["home_win", "draw", "away_win"]
    model_probability: float
    market_probability: float | None = None
    odds: float | None = None
    edge: float | None = None
    kelly_fraction: float = 0
    stake: float = 0
    rationale: str


class PredictionResult(BaseModel):
    match: MatchPredictionRequest
    probabilities: OutcomeProbability
    expected_score: tuple[float, float]
    most_likely_score: str
    odds: OddsRecord | None = None
    bet_signals: list[BetSignal]
    agent_findings: list[AgentFinding]
    explanation: str
    shap_values: dict[str, float] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class TournamentSimulationRequest(BaseModel):
    teams: list[str] = Field(min_length=2)
    simulations: int = Field(default=10000, ge=100, le=100000)
    fixtures: list["TournamentFixture"] = Field(default_factory=list)
    standings: list["GroupStandingRecord"] = Field(default_factory=list)
    use_current_state: bool = False


class TournamentFixture(BaseModel):
    match_id: str
    home_team: str
    away_team: str
    group_name: str | None = None
    round_name: str = "小组赛"
    kickoff_time: datetime | None = None
    completed: bool = False
    home_score: int | None = None
    away_score: int | None = None
    source: str = "manual"


class TournamentTeamProbability(BaseModel):
    team: str
    group_advance: float
    quarter_final: float
    semi_final: float
    final: float
    champion: float


class GroupStandingRecord(BaseModel):
    group_name: str
    rank: int
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    goal_difference: int = 0
    points: int = 0
    source: str = "espn"
    source_url: str = ""
    scraped_at: datetime = Field(default_factory=datetime.utcnow)


class ScrapeResult(BaseModel):
    provider: str
    source_url: str
    scraped_at: datetime
    record_count: int
    snapshot_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    records: list[OddsRecord] = Field(default_factory=list)


class HealthStatus(BaseModel):
    status: str
    db_path: str
    odds_records: int
    match_records: int
    live_matches: int
    standing_records: int = 0
    llm_enabled: bool
    model_version: str
    scheduler_running: bool


class WorldCupDataImportResult(BaseModel):
    status: str
    source_url: str = ""
    raw_path: str = ""
    output_path: str
    rows: int
    years: list[int]
    imported_records: int = 0
    teams_with_elo: int = 0
    message: str = ""


class BacktestMetrics(BaseModel):
    tournament_year: int
    brier: float
    log_loss: float
    roi: float
    max_drawdown: float
    record_count: int
    baseline_brier: float
    baseline_log_loss: float


class BacktestRunRequest(BaseModel):
    years: list[int] = Field(default_factory=lambda: [2014, 2018, 2022])
    edge_threshold: float = Field(default=0.025, ge=-0.2, le=0.5)
    kelly_fraction: float = Field(default=0.25, ge=0, le=1)
    max_stake_fraction: float = Field(default=0.05, ge=0, le=0.5)
    initial_bankroll: float = Field(default=1000, gt=0)
    home_odds: float = Field(default=2.5, gt=1)
    draw_odds: float = Field(default=3.1, gt=1)
    away_odds: float = Field(default=2.5, gt=1)


class BacktestRunResult(BaseModel):
    id: str
    run_at: datetime
    metrics: list[BacktestMetrics]
    params: BacktestRunRequest = Field(default_factory=BacktestRunRequest)


class TournamentBracket(BaseModel):
    teams: list[str]
    simulations: int
    results: list[TournamentTeamProbability]
    group_results: dict[str, Any] = Field(default_factory=dict)
    fixtures_used: int = 0
    current_state_used: bool = False


class TournamentScheduleImport(BaseModel):
    teams: list[str]
    fixtures: list[TournamentFixture]
    standings: list[GroupStandingRecord] = Field(default_factory=list)
    imported_at: datetime = Field(default_factory=datetime.utcnow)
