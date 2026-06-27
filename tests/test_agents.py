from __future__ import annotations

from worldcup_forecast.agents import (
    BullBearDebateAgents,
    FormAgent,
    NewsSentimentAgent,
    OddsMarketAgent,
    RiskManagerAgent,
    StrengthAgent,
    build_agent_findings,
    template_explanation,
)
from worldcup_forecast.modeling import BaselineForecastModel, build_bet_signals
from worldcup_forecast.schemas import AgentFinding, MatchPredictionRequest, OddsRecord, PredictionResult


def _sample_request() -> MatchPredictionRequest:
    return MatchPredictionRequest(home_team="Brazil", away_team="Germany", bankroll=1000)


def _sample_prediction() -> PredictionResult:
    req = _sample_request()
    dist = BaselineForecastModel().predict_score_distribution(req)
    return PredictionResult(
        match=req,
        probabilities=dist.probabilities,
        expected_score=(dist.expected_home_goals, dist.expected_away_goals),
        most_likely_score=dist.most_likely_score,
        bet_signals=build_bet_signals(req, dist.probabilities, None),
        agent_findings=[],
        explanation="",
    )


def _assert_finding(finding: AgentFinding) -> None:
    assert 0.0 <= finding.confidence <= 1.0
    assert finding.signal in ("positive", "neutral", "negative")
    assert isinstance(finding.rationale, str) and len(finding.rationale) > 0
    assert isinstance(finding.sources, list)


def test_strength_agent_schema():
    finding = StrengthAgent().analyze(_sample_request())
    _assert_finding(finding)
    assert "elo" in finding.metrics or "home_elo" in finding.metrics


def test_form_agent_schema():
    _assert_finding(FormAgent().analyze(_sample_request()))


def test_form_agent_uses_real_intl_data(tmp_store):
    from datetime import date

    rows = [
        {"date": date(2024, 1, i + 1), "home_team": "Brazil", "away_team": "Chile",
         "home_score": 3, "away_score": 0, "tournament": "Friendly", "neutral": False}
        for i in range(6)
    ] + [
        {"date": date(2024, 2, i + 1), "home_team": "Chile", "away_team": "Peru",
         "home_score": 0, "away_score": 2, "tournament": "Friendly", "neutral": False}
        for i in range(6)
    ]
    tmp_store.insert_intl_results(rows)
    finding = FormAgent(tmp_store).analyze(
        MatchPredictionRequest(home_team="Brazil", away_team="Chile")
    )
    _assert_finding(finding)
    # Brazil winning every game should read as positive recent form.
    assert finding.signal == "positive"
    assert finding.metrics["home_ppg"] > finding.metrics["away_ppg"]
    assert finding.sources == ["internal:intl_results"]


def test_news_agent_honest_when_search_unconfigured():
    """With no search source configured, the news agent must report search_ok
    False rather than silently claiming a neutral signal."""
    from worldcup_forecast.schemas import SearchSettings

    finding = NewsSentimentAgent(SearchSettings(provider="none", enabled=False)).analyze(
        _sample_request()
    )
    _assert_finding(finding)
    assert finding.metrics.get("search_ok") is False
    assert "搜索源" in finding.rationale or "RSS" in finding.rationale


def test_strength_agent_uses_team_elo(tmp_store):
    tmp_store.upsert_team_elo({"Atlantis": 2200.0, "Lilliput": 1400.0})
    finding = StrengthAgent(tmp_store).analyze(
        MatchPredictionRequest(home_team="Atlantis", away_team="Lilliput")
    )
    _assert_finding(finding)
    assert finding.signal == "positive"
    assert finding.metrics["home_elo"] == 2200.0
    assert finding.sources == ["internal:team_elo"]


def test_news_sentiment_agent_schema():
    _assert_finding(NewsSentimentAgent().analyze(_sample_request()))


def test_odds_market_agent_no_odds():
    finding = OddsMarketAgent().analyze(None)
    _assert_finding(finding)
    assert finding.signal == "neutral"


def test_odds_market_agent_with_odds():
    odds = OddsRecord(
        match_id="t1",
        home_team="Brazil",
        away_team="Germany",
        play_type="胜平负",
        win_odds=2.4,
        draw_odds=3.2,
        lose_odds=2.9,
        source_url="https://trade.500.com/jczq/",
    )
    finding = OddsMarketAgent().analyze(odds)
    _assert_finding(finding)
    assert finding.metrics.get("win_odds") == 2.4


def test_bull_bear_debate_agents():
    findings = BullBearDebateAgents().debate(_sample_prediction())
    assert len(findings) == 2
    agents = {f.agent for f in findings}
    assert "正方研究员" in agents
    assert "反方研究员" in agents
    for finding in findings:
        _assert_finding(finding)


def test_risk_manager_agent():
    finding = RiskManagerAgent().analyze(_sample_prediction())
    _assert_finding(finding)
    assert "max_stake" in finding.metrics


def test_build_agent_findings_count():
    findings = build_agent_findings(_sample_request(), None)
    assert len(findings) == 4
    for finding in findings:
        _assert_finding(finding)


def test_template_explanation_contains_team_names():
    text = template_explanation(_sample_prediction())
    assert "Brazil" in text
    assert "Germany" in text
