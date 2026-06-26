from worldcup_forecast.modeling import BaselineForecastModel, build_bet_signals
from worldcup_forecast.schemas import MatchPredictionRequest, OddsRecord


def test_baseline_probabilities_sum_to_one():
    request = MatchPredictionRequest(home_team="Brazil", away_team="Germany")
    distribution = BaselineForecastModel().predict_score_distribution(request)
    total = (
        distribution.probabilities.home_win
        + distribution.probabilities.draw
        + distribution.probabilities.away_win
    )

    assert round(total, 6) == 1
    assert distribution.most_likely_score


def test_research_parameters_change_prediction():
    base = MatchPredictionRequest(home_team="Brazil", away_team="Germany")
    tuned = MatchPredictionRequest(
        home_team="Brazil",
        away_team="Germany",
        neutral_site=False,
        home_advantage_elo=120,
        draw_bias=1.3,
        goal_rate_multiplier=1.15,
    )
    base_dist = BaselineForecastModel().predict_score_distribution(base)
    tuned_dist = BaselineForecastModel().predict_score_distribution(tuned)

    assert base_dist.probabilities != tuned_dist.probabilities


def test_bet_signals_include_kelly_and_edge():
    request = MatchPredictionRequest(home_team="Brazil", away_team="Germany", bankroll=1000)
    distribution = BaselineForecastModel().predict_score_distribution(request)
    odds = OddsRecord(
        match_id="demo",
        home_team="Brazil",
        away_team="Germany",
        play_type="胜平负",
        win_odds=2.4,
        draw_odds=3.2,
        lose_odds=2.9,
        source_url="https://trade.500.com/jczq/",
    )
    signals = build_bet_signals(request, distribution.probabilities, odds)

    assert len(signals) == 3
    assert all(signal.kelly_fraction >= 0 for signal in signals)
    assert all(signal.edge is not None for signal in signals)
