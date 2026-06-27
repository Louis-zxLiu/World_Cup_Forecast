from __future__ import annotations

import asyncio
import json
import random
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from worldcup_forecast.agents import (
    BullBearDebateAgents,
    RiskManagerAgent,
    build_agent_findings,
    template_explanation,
)
from worldcup_forecast.backtest import BacktestRunner
from worldcup_forecast.config import get_config
from worldcup_forecast.espn import ESPNProvider
from worldcup_forecast.graph import build_forecast_graph
from worldcup_forecast.ingest import (
    EloUpdater,
    HistoricalIngestor,
    InternationalResultsDownloader,
    InternationalResultsIngestor,
    WorldCupDataDownloader,
)
from worldcup_forecast.llm import OpenAICompatibleClient, public_llm_settings, public_search_settings
from worldcup_forecast.ml_model import SHAPExplainer, XGBoostForecaster
from worldcup_forecast.modeling import BaselineForecastModel, build_bet_signals
from worldcup_forecast.odds import FiveHundredLotteryProvider
from worldcup_forecast.odds_cleaner import OddsCleaner
from worldcup_forecast.reasoning import reason_for_finding
from worldcup_forecast.search import WebSearchProvider
from worldcup_forecast.schemas import (
    AskRequest,
    AskResponse,
    BacktestRunResult,
    BacktestRunRequest,
    GroupStandingRecord,
    HealthStatus,
    LLMSettings,
    MatchPredictionRequest,
    PredictionResult,
    ScrapeResult,
    SearchSettings,
    TournamentBracket,
    TournamentFixture,
    TournamentScheduleImport,
    TournamentSimulationRequest,
    WorldCupDataImportResult,
)
from worldcup_forecast.storage import ForecastStore
from worldcup_forecast.tournament import WorldCupSimulator

_scheduler_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler_running
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        scheduler = AsyncIOScheduler()

        async def _scrape_job():
            try:
                local_store = ForecastStore()
                result = await FiveHundredLotteryProvider().scrape()
                cleaned, _warnings = await OddsCleaner(local_store.get_llm_settings()).clean(result.records)
                local_store.insert_odds(cleaned)
            except Exception:
                pass

        async def _espn_job():
            try:
                local_store = ForecastStore()
                provider = ESPNProvider()
                matches = await provider.fetch_all_matches()
                local_store.upsert_live_matches([m.to_dict() for m in matches])
            except Exception:
                pass

        def _ingest_job():
            try:
                local_store = ForecastStore()
                csv_path = Path("data/wc_results.csv")
                if csv_path.exists():
                    HistoricalIngestor(local_store).load_csv(csv_path)
                    EloUpdater(local_store).update_from_results()
                    baseline_model.refresh_elo(local_store)
            except Exception:
                pass

        scheduler.add_job(_scrape_job, "cron", minute=0)
        scheduler.add_job(_espn_job, "interval", minutes=5)
        scheduler.add_job(_ingest_job, "cron", hour=6, minute=0)
        scheduler.start()
        _scheduler_running = True
    except Exception:
        _scheduler_running = False

    yield
    _scheduler_running = False


app = FastAPI(title="World Cup Forecast API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=(
        r"^https?://("
        r"localhost|127\.0\.0\.1|\[::1\]|"
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"192\.168\.\d{1,3}\.\d{1,3}|"
        r"172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
        r")(:\d+)?$"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = ForecastStore()
baseline_model = BaselineForecastModel(store=store)
xgb_model = XGBoostForecaster(store)
xgb_model.load()


def _active_model():
    if xgb_model._clf is not None:
        return xgb_model
    return baseline_model


def _parse_match_datetime(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fixtures_from_live_matches(
    matches: list[dict],
    standings: list[GroupStandingRecord],
) -> list[TournamentFixture]:
    team_to_group = {row.team: row.group_name for row in standings}
    fixtures: list[TournamentFixture] = []
    for match in matches:
        home = match.get("home_team") or ""
        away = match.get("away_team") or ""
        if not home or not away:
            continue
        home_group = team_to_group.get(home)
        away_group = team_to_group.get(away)
        group_name = home_group if home_group and home_group == away_group else None
        round_name = "小组赛" if group_name else (match.get("stage") or "淘汰赛")
        fixtures.append(
            TournamentFixture(
                match_id=str(match.get("match_id") or f"{home}-{away}"),
                home_team=home,
                away_team=away,
                group_name=group_name,
                round_name=str(round_name),
                kickoff_time=_parse_match_datetime(match.get("date")),
                completed=bool(match.get("completed")),
                home_score=match.get("home_score"),
                away_score=match.get("away_score"),
                source="espn",
            )
        )
    return fixtures


def _teams_from_state(
    teams: list[str],
    fixtures: list[TournamentFixture],
    standings: list[GroupStandingRecord],
) -> list[str]:
    ordered: list[str] = []
    for team in teams:
        if team and team not in ordered:
            ordered.append(team)
    for row in standings:
        if row.team and row.team not in ordered:
            ordered.append(row.team)
    for fixture in fixtures:
        for team in (fixture.home_team, fixture.away_team):
            if team and team not in ordered:
                ordered.append(team)
    return ordered


@app.get("/api/health", response_model=HealthStatus)
def health() -> HealthStatus:
    settings = store.get_llm_settings()
    active = _active_model()
    return HealthStatus(
        status="ok",
        db_path=str(get_config().db_path),
        odds_records=store.odds_count(),
        match_records=store.match_count(),
        live_matches=store.live_match_count(),
        standing_records=store.standing_count(),
        llm_enabled=settings.enabled,
        model_version=getattr(active, "version", baseline_model.version),
        scheduler_running=_scheduler_running,
    )


@app.get("/api/settings/llm")
def get_llm_settings():
    return public_llm_settings(store.get_llm_settings())


@app.put("/api/settings/llm")
def update_llm_settings(settings: LLMSettings):
    if not settings.api_key:
        settings.api_key = store.get_llm_settings().api_key
    saved = store.save_llm_settings(settings)
    return public_llm_settings(saved)


@app.post("/api/settings/llm/test")
async def test_llm_settings(settings: LLMSettings | None = None):
    candidate = settings or store.get_llm_settings()
    try:
        return await OpenAICompatibleClient(candidate).test_connection()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/settings/search")
def get_search_settings():
    return public_search_settings(store.get_search_settings())


@app.put("/api/settings/search")
def update_search_settings(settings: SearchSettings):
    if not settings.api_key:
        settings.api_key = store.get_search_settings().api_key
    saved = store.save_search_settings(settings)
    return public_search_settings(saved)


@app.post("/api/settings/search/test")
async def test_search_settings(settings: SearchSettings | None = None):
    candidate = settings or store.get_search_settings()
    if candidate and not candidate.api_key:
        candidate = candidate.model_copy(update={"api_key": store.get_search_settings().api_key})
    outcome = await WebSearchProvider(candidate).search("世界杯 足球 最新")
    if not outcome.ok:
        raise HTTPException(status_code=400, detail=outcome.error)
    return {"ok": True, "results": len(outcome.hits), "sample": [h.title for h in outcome.hits[:3]]}


@app.post("/api/odds/china-lottery/scrape", response_model=ScrapeResult)
async def scrape_china_lottery_odds(use_playwright: bool = False):
    result = await FiveHundredLotteryProvider().scrape(use_playwright=use_playwright)
    cleaned_records, cleaning_warnings = await OddsCleaner(store.get_llm_settings()).clean(
        result.records
    )
    result.records = cleaned_records
    result.record_count = len(cleaned_records)
    result.warnings.extend(cleaning_warnings)
    store.insert_odds(result.records)
    return result


@app.get("/api/odds/latest")
def latest_odds(limit: int = 100):
    return store.latest_odds(limit=limit)


@app.post("/api/ingest/run")
def run_ingestion(force_download: bool = True, include_international: bool = True) -> WorldCupDataImportResult:
    try:
        result = WorldCupDataDownloader().download_and_prepare(force=force_download)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    count = HistoricalIngestor(store).load_csv(Path(result.output_path))
    # Pull the full international archive so Elo and recent form cover every
    # national team, not just frequent World Cup participants.
    if include_international:
        try:
            intl_path = InternationalResultsDownloader().download(force=force_download)
            intl_count = InternationalResultsIngestor(store).load_csv(intl_path)
            result.message = f"{result.message} 国际比赛档案 {intl_count} 场已载入。".strip()
        except Exception as exc:  # noqa: BLE001 - international data is best-effort
            result.message = f"{result.message} 国际比赛档案下载失败：{exc}".strip()
    elo = EloUpdater(store).update_from_results()
    baseline_model.refresh_elo(store)
    result.imported_records = count
    result.teams_with_elo = len(elo)
    return result


@app.post("/api/models/train")
def train_model():
    try:
        version = xgb_model.train()
        return {"status": "ok", "model_version": version}
    except ValueError as exc:
        return {"status": "skipped", "reason": str(exc), "model_version": baseline_model.version}


@app.post("/api/predict/match", response_model=PredictionResult)
async def predict_match(request: MatchPredictionRequest):
    llm_settings = store.get_llm_settings()
    search_settings = store.get_search_settings()
    graph = build_forecast_graph(store, search_settings)
    initial_state = {
        "request": request.model_dump(),
        "llm_settings": llm_settings.model_dump(),
        "search_settings": search_settings.model_dump(),
        "agent_findings": [],
    }
    final_state = await graph.ainvoke(initial_state)

    from worldcup_forecast.schemas import (
        AgentFinding, BetSignal, OutcomeProbability, OddsRecord,
    )
    probs = OutcomeProbability(**final_state["probabilities"])
    signals = [BetSignal(**s) for s in final_state.get("bet_signals", [])]
    findings = []
    for f in final_state.get("agent_findings", []):
        try:
            findings.append(AgentFinding(**f))
        except Exception:
            pass
    odds_raw = final_state.get("odds")
    odds = OddsRecord(**odds_raw) if odds_raw else None

    result = PredictionResult(
        match=request,
        probabilities=probs,
        expected_score=final_state.get("expected_score", (1.3, 1.1)),
        most_likely_score=final_state.get("most_likely_score", "1-1"),
        odds=odds,
        bet_signals=signals,
        agent_findings=findings,
        shap_values=final_state.get("shap_values", {}),
        explanation=final_state.get("explanation", ""),
    )
    # report_node already persisted the report; only save if it somehow wasn't set.
    report_id = final_state.get("report_id")
    if not report_id:
        report_id = str(uuid.uuid4())
        store.save_report(report_id, result.model_dump_json())
    return result


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@app.post("/api/predict/match/stream")
async def predict_match_stream(request: MatchPredictionRequest):
    """Stream the multi-agent prediction via LangGraph, exposing each agent's reasoning live.

    Emits Server-Sent Events: ``prediction`` (after supervisor), one ``reasoning``
    per agent node completion, then ``report`` and ``done``.
    """
    # Node names that emit a reasoning SSE event when they complete.
    _ANALYSIS_NODES = {"strength_node", "form_node", "news_node", "odds_node"}
    _ALL_AGENT_NODES = _ANALYSIS_NODES | {"debate_node", "risk_node"}

    async def event_stream():
        llm_settings = store.get_llm_settings()
        search_settings = store.get_search_settings()
        graph = build_forecast_graph(store, search_settings)
        initial_state = {
            "request": request.model_dump(),
            "llm_settings": llm_settings.model_dump(),
            "search_settings": search_settings.model_dump(),
            "agent_findings": [],
        }

        _ALL_NODES = _ANALYSIS_NODES | _ALL_AGENT_NODES | {"supervisor", "report_node"}

        prediction_sent = False

        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            # Node start → graph visualization
            if kind == "on_chain_start" and name in _ALL_NODES:
                yield _sse("node_start", {"node": name})

            # Supervisor end → prediction data + node_end
            elif kind == "on_chain_end" and name == "supervisor" and not prediction_sent:
                output = event.get("data", {}).get("output", {})
                yield _sse("prediction", {
                    "probabilities": output.get("probabilities", {}),
                    "expected_score": output.get("expected_score", []),
                    "most_likely_score": output.get("most_likely_score", ""),
                    "bet_signals": output.get("bet_signals", []),
                    "odds": output.get("odds"),
                    "shap_values": output.get("shap_values", {}),
                })
                prediction_sent = True
                yield _sse("node_end", {"node": "supervisor"})

            # Analysis/debate/risk node end → reasoning trace + node_end
            elif kind == "on_chain_end" and name in _ALL_AGENT_NODES:
                output = event.get("data", {}).get("output", {})
                new_findings = output.get("agent_findings", [])
                for finding in new_findings:
                    from worldcup_forecast.schemas import AgentFinding
                    from worldcup_forecast.reasoning import deterministic_steps
                    try:
                        af = AgentFinding(**finding)
                    except Exception:
                        continue
                    steps = deterministic_steps(af)
                    from worldcup_forecast.schemas import AgentReasoning
                    reasoning = AgentReasoning(
                        agent=af.agent,
                        confidence=af.confidence,
                        signal=af.signal,
                        steps=steps,
                        rationale=af.rationale,
                        sources=af.sources,
                        metrics=af.metrics,
                        powered_by="llm" if llm_settings.enabled and llm_settings.api_key else "deterministic",
                    )
                    yield _sse("reasoning", reasoning.model_dump())
                yield _sse("node_end", {"node": name})

            # Report node end → report + done + node_end
            elif kind == "on_chain_end" and name == "report_node":
                output = event.get("data", {}).get("output", {})
                report_id = output.get("report_id", str(uuid.uuid4()))
                explanation = output.get("explanation", "")
                yield _sse("node_end", {"node": "report_node"})
                yield _sse("report", {"explanation": explanation, "report_id": report_id})
                yield _sse("done", {"report_id": report_id})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/predict/tournament", response_model=TournamentBracket)
def predict_tournament(request: TournamentSimulationRequest):
    random.seed(42)
    elo_map = {**store.get_team_elo()}
    standings = request.standings
    fixtures = request.fixtures
    if request.use_current_state and not standings:
        standings = store.get_group_standings()
    if request.use_current_state and not fixtures:
        fixtures = _fixtures_from_live_matches(store.get_live_matches(), standings)
    teams = _teams_from_state(request.teams, fixtures, standings)
    sim = WorldCupSimulator(
        teams,
        request.simulations,
        elo_map,
        fixtures=fixtures,
        standings=standings,
        use_current_state=request.use_current_state,
    )
    return sim.simulate()


@app.post("/api/tournament/schedule/import", response_model=TournamentScheduleImport)
async def import_tournament_schedule():
    provider = ESPNProvider()
    matches = await provider.fetch_all_matches()
    standings = await provider.fetch_standings()
    match_dicts = [m.to_dict() for m in matches]
    store.upsert_live_matches(match_dicts)
    if standings:
        store.replace_group_standings(standings)
    fixtures = _fixtures_from_live_matches(match_dicts, standings)
    teams = _teams_from_state([], fixtures, standings)
    return TournamentScheduleImport(teams=teams, fixtures=fixtures, standings=standings)


@app.post("/api/espn/sync")
async def espn_sync():
    provider = ESPNProvider()
    matches = await provider.fetch_all_matches()
    teams = await provider.fetch_teams()
    store.upsert_live_matches([m.to_dict() for m in matches])
    store.upsert_espn_teams(teams)
    return {"status": "ok", "matches_synced": len(matches), "teams_synced": len(teams)}


@app.get("/api/espn/matches")
def espn_matches():
    return store.get_live_matches()


def _quick_prediction(request: MatchPredictionRequest) -> PredictionResult:
    """Run a non-streaming prediction without LLM narration (fast, for batch use)."""
    active = _active_model()
    odds = store.find_match_odds(request.home_team, request.away_team)
    distribution = active.predict_score_distribution(request)
    result = PredictionResult(
        match=request,
        probabilities=distribution.probabilities,
        expected_score=(
            round(distribution.expected_home_goals, 3),
            round(distribution.expected_away_goals, 3),
        ),
        most_likely_score=distribution.most_likely_score,
        odds=odds,
        bet_signals=build_bet_signals(request, distribution.probabilities, odds),
        agent_findings=build_agent_findings(request, odds, store),
        explanation="",
        score_matrix=distribution.score_matrix,
    )
    result.agent_findings.extend(BullBearDebateAgents().debate(result))
    result.agent_findings.append(RiskManagerAgent().analyze(result))
    result.explanation = template_explanation(result)
    return result


@app.post("/api/predict/match/quick", response_model=PredictionResult)
def predict_match_quick(request: MatchPredictionRequest):
    """Fast deterministic prediction (no LLM, no LangGraph). For batch/programmatic use."""
    return _quick_prediction(request)


@app.get("/api/predict/today")
def predict_today(bankroll: float = 10000, limit: int = 12):
    """One-click overview: predict every upcoming/live match on the schedule.

    Requires no input from the user. Returns each match's probabilities plus its
    single strongest value bet (if any), sorted so the best opportunities surface
    first.
    """
    matches = store.get_live_matches()
    upcoming = [
        m for m in matches
        if m.get("home_team") and m.get("away_team") and not m.get("completed")
    ]
    if not upcoming:
        upcoming = [m for m in matches if m.get("home_team") and m.get("away_team")]

    cards = []
    for match in upcoming[:limit]:
        request = MatchPredictionRequest(
            home_team=match["home_team"],
            away_team=match["away_team"],
            neutral_site=True,
            bankroll=bankroll,
        )
        result = _quick_prediction(request)
        value_bets = [s for s in result.bet_signals if s.stake > 0 and (s.edge or 0) > request.value_edge_threshold]
        best = max(value_bets, key=lambda s: s.edge or 0, default=None)
        cards.append(
            {
                "match_id": match.get("match_id"),
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "kickoff_time": match.get("date"),
                "status": match.get("status_name"),
                "probabilities": result.probabilities.model_dump(),
                "most_likely_score": result.most_likely_score,
                "recommendation": best.model_dump() if best else None,
                "has_value": best is not None,
            }
        )
    cards.sort(key=lambda c: (c["recommendation"]["edge"] if c["recommendation"] else -1), reverse=True)
    return {"count": len(cards), "cards": cards}


def _extract_teams_rule(question: str) -> tuple[str | None, str | None]:
    """Find up to two team names in a question using the known alias map."""
    from worldcup_forecast.odds import TEAM_ALIASES

    found: list[str] = []
    # Chinese aliases first (longer names matched before shorter substrings).
    for alias in sorted(TEAM_ALIASES, key=len, reverse=True):
        if alias in question and TEAM_ALIASES[alias] not in found:
            found.append(TEAM_ALIASES[alias])
        if len(found) == 2:
            break
    if len(found) < 2:
        known_en = set(store.get_team_elo().keys())
        for team in sorted(known_en, key=len, reverse=True):
            if team.lower() in question.lower() and team not in found:
                found.append(team)
            if len(found) == 2:
                break
    home = found[0] if found else None
    away = found[1] if len(found) > 1 else None
    return home, away


async def _extract_teams_llm(question: str, settings: LLMSettings) -> tuple[str | None, str | None]:
    prompt = (
        "从这句话里提取两支足球国家队的英文规范名（如 Brazil、Germany）。"
        "第一支按主队、第二支按客队。只返回 JSON："
        '{"home":"...","away":"..."}，无法识别填 null。\n'
        f"句子：{question}"
    )
    raw = await OpenAICompatibleClient(settings).complete(
        "你是足球比赛信息抽取器，只输出 JSON。", prompt
    )
    cleaned = raw.strip().strip("`")
    start, end = cleaned.find("{"), cleaned.rfind("}")
    data = json.loads(cleaned[start : end + 1])
    return data.get("home"), data.get("away")


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Natural-language entry point: ask a question, get a prediction-backed answer."""
    settings = store.get_llm_settings()
    home = away = None
    if settings.enabled and settings.api_key:
        try:
            home, away = await _extract_teams_llm(request.question, settings)
        except Exception:
            home, away = None, None
    if not (home and away):
        home, away = _extract_teams_rule(request.question)

    if not (home and away):
        return AskResponse(
            question=request.question,
            answer="没能从问题里识别出两支球队。可以试试「巴西对德国谁会赢」这样的问法。",
            matched=False,
        )

    pred = _quick_prediction(
        MatchPredictionRequest(home_team=home, away_team=away, bankroll=request.bankroll)
    )
    probs = pred.probabilities
    outcome_label = {"home_win": f"{home}胜", "draw": "平局", "away_win": f"{away}胜"}
    best_key = max(("home_win", "draw", "away_win"), key=lambda k: getattr(probs, k))
    answer = (
        f"{home} vs {away}：胜平负概率 "
        f"{probs.home_win:.0%}/{probs.draw:.0%}/{probs.away_win:.0%}，"
        f"最可能结果是{outcome_label[best_key]}，预计比分 {pred.most_likely_score}。"
    )
    value_bets = [s for s in pred.bet_signals if s.stake > 0 and (s.edge or 0) > 0.025]
    if value_bets:
        best = max(value_bets, key=lambda s: s.edge or 0)
        answer += f" 价值提示：{outcome_label[best.outcome]} 有 {best.edge:.1%} 正向 edge，建议小仓位 {best.stake:.0f}。"
    else:
        answer += " 当前没有明显的价值投注机会，建议观望。"

    return AskResponse(
        question=request.question,
        home_team=home,
        away_team=away,
        answer=answer,
        matched=True,
    )


@app.get("/api/espn/teams")
def espn_teams():
    return store.get_espn_teams()


@app.post("/api/standings/sync")
async def standings_sync():
    provider = ESPNProvider()
    records = await provider.fetch_standings()
    saved = store.replace_group_standings(records)
    return {"status": "ok", "standing_records": saved}


@app.get("/api/standings/latest")
def latest_standings():
    return store.get_group_standings()


def _backtest_template_explanation(result: BacktestRunResult) -> str:
    if not result.metrics:
        return "回测未找到历史比赛记录，请先导入数据后再运行。"
    years = "/".join(str(m.tournament_year) for m in result.metrics)
    avg_brier = sum(m.brier for m in result.metrics) / len(result.metrics)
    avg_baseline = sum(m.baseline_brier for m in result.metrics) / len(result.metrics)
    avg_roi = sum(m.roi for m in result.metrics) / len(result.metrics)
    avg_dd = sum(m.max_drawdown for m in result.metrics) / len(result.metrics)
    better = "优于" if avg_brier < avg_baseline else "不及"
    roi_str = f"+{avg_roi*100:.1f}%" if avg_roi >= 0 else f"{avg_roi*100:.1f}%"
    return (
        f"模型在 {years} 届世界杯回测中，平均 Brier 分 {avg_brier:.3f}（越低越准），"
        f"{better}随机基准 {avg_baseline:.3f}。"
        f"模拟投注 ROI 为 {roi_str}，最大回撤 {avg_dd*100:.1f}%。"
        f"{'整体预测能力高于随机猜测，模型有效。' if avg_brier < avg_baseline else '模型尚未显著超越基准，可尝试调整参数或补充更多数据。'}"
    )


@app.post("/api/backtest/run")
async def run_backtest(request: BacktestRunRequest | None = None):
    if store.match_count() == 0:
        try:
            data_result = WorldCupDataDownloader().download_and_prepare(force=True)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        HistoricalIngestor(store).load_csv(Path(data_result.output_path))
        EloUpdater(store).update_from_results()
    runner = BacktestRunner(store)
    result = runner.run(params=request or BacktestRunRequest())

    explanation = _backtest_template_explanation(result)
    llm_settings = store.get_llm_settings()
    if llm_settings.enabled and llm_settings.api_key:
        try:
            explanation = await OpenAICompatibleClient(llm_settings).complete(
                "你是足球数据分析师，请用通俗中文向普通用户解释这次回测结果，说明模型是否有效、投注表现如何，不超过120字，不要出现专业术语缩写。",
                f"回测结果：{result.model_dump_json()}",
            )
        except Exception:
            pass

    return {**result.model_dump(), "explanation": explanation}


@app.get("/api/backtest/results")
def list_backtest_results():
    return store.list_backtest_runs()


@app.get("/api/reports/{report_id}")
def get_report(report_id: str):
    report = store.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="report not found")
    return report


@app.get("/api/matches")
def matches():
    results = store.get_match_results()
    if results:
        return [
            {
                "match_id": f"{r['year']}-{r['stage']}-{r['home_team']}-{r['away_team']}",
                "kickoff_time": r["date"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "competition": f"{r['tournament']} {r['year']}",
            }
            for r in results[-20:]
        ]
    return [
        {
            "match_id": "demo-001",
            "kickoff_time": datetime.utcnow(),
            "home_team": "Brazil",
            "away_team": "Germany",
            "competition": "World Cup demo",
        },
        {
            "match_id": "demo-002",
            "kickoff_time": datetime.utcnow(),
            "home_team": "Argentina",
            "away_team": "France",
            "competition": "World Cup demo",
        },
    ]
