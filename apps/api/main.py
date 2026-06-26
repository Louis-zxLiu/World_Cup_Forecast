from __future__ import annotations

import random
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from worldcup_forecast.agents import (
    BullBearDebateAgents,
    RiskManagerAgent,
    build_agent_findings,
    template_explanation,
)
from worldcup_forecast.backtest import BacktestRunner
from worldcup_forecast.config import get_config
from worldcup_forecast.espn import ESPNProvider
from worldcup_forecast.ingest import EloUpdater, HistoricalIngestor, WorldCupDataDownloader
from worldcup_forecast.llm import OpenAICompatibleClient, public_llm_settings
from worldcup_forecast.ml_model import SHAPExplainer, XGBoostForecaster
from worldcup_forecast.modeling import BaselineForecastModel, build_bet_signals
from worldcup_forecast.odds import FiveHundredLotteryProvider
from worldcup_forecast.odds_cleaner import OddsCleaner
from worldcup_forecast.schemas import (
    BacktestRunResult,
    BacktestRunRequest,
    GroupStandingRecord,
    HealthStatus,
    LLMSettings,
    MatchPredictionRequest,
    PredictionResult,
    ScrapeResult,
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
baseline_model = BaselineForecastModel()
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
def run_ingestion(force_download: bool = True) -> WorldCupDataImportResult:
    try:
        result = WorldCupDataDownloader().download_and_prepare(force=force_download)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    count = HistoricalIngestor(store).load_csv(Path(result.output_path))
    elo = EloUpdater(store).update_from_results()
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
    active = _active_model()
    odds = store.find_match_odds(request.home_team, request.away_team)
    distribution = active.predict_score_distribution(request)
    shap_values: dict[str, float] = {}
    if xgb_model._clf is not None:
        shap_values = SHAPExplainer(xgb_model).top_features(
            request.home_team, request.away_team, request.neutral_site
        )

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
        agent_findings=build_agent_findings(request, odds),
        shap_values=shap_values,
        explanation="",
    )
    result.agent_findings.extend(BullBearDebateAgents().debate(result))
    result.agent_findings.append(RiskManagerAgent().analyze(result))

    llm_settings = store.get_llm_settings()
    if llm_settings.enabled and llm_settings.api_key:
        try:
            prompt = (
                "请用中文生成一份简洁、可解释、带风险提示的世界杯单场预测报告。"
                f"数据如下：{result.model_dump_json()}"
            )
            result.explanation = await OpenAICompatibleClient(llm_settings).complete(
                "你是谨慎的足球量化和投注风控分析师，不承诺稳赚。",
                prompt,
            )
        except Exception:
            result.explanation = template_explanation(result)
    else:
        result.explanation = template_explanation(result)

    report_id = str(uuid.uuid4())
    store.save_report(report_id, result.model_dump_json())
    return result


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


@app.post("/api/backtest/run", response_model=BacktestRunResult)
def run_backtest(request: BacktestRunRequest | None = None):
    if store.match_count() == 0:
        try:
            data_result = WorldCupDataDownloader().download_and_prepare(force=True)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        HistoricalIngestor(store).load_csv(Path(data_result.output_path))
        EloUpdater(store).update_from_results()
    runner = BacktestRunner(store)
    return runner.run(params=request or BacktestRunRequest())


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
