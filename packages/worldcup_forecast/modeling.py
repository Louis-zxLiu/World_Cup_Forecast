from __future__ import annotations

import math
from dataclasses import dataclass

from .odds import implied_probabilities
from .schemas import BetSignal, MatchPredictionRequest, OddsRecord, OutcomeProbability

BASE_ELO = {
    "Brazil": 2095,
    "Argentina": 2088,
    "France": 2055,
    "England": 2020,
    "Spain": 2015,
    "Germany": 1980,
    "Portugal": 1970,
    "Netherlands": 1965,
    "Italy": 1940,
    "United States": 1780,
    "Japan": 1810,
    "China": 1500,
}


def team_strength(team: str) -> float:
    return BASE_ELO.get(team, 1700)


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def normalize(values: dict[str, float]) -> OutcomeProbability:
    total = sum(max(v, 0.001) for v in values.values())
    return OutcomeProbability(
        home_win=max(values["home_win"], 0.001) / total,
        draw=max(values["draw"], 0.001) / total,
        away_win=max(values["away_win"], 0.001) / total,
    )


def poisson_probability(lam: float, goals: int) -> float:
    return math.exp(-lam) * lam**goals / math.factorial(goals)


@dataclass
class ScoreDistribution:
    probabilities: OutcomeProbability
    expected_home_goals: float
    expected_away_goals: float
    most_likely_score: str


class BaselineForecastModel:
    version = "baseline-elo-poisson-v0.1"

    def predict_score_distribution(self, request: MatchPredictionRequest) -> ScoreDistribution:
        home_elo = team_strength(request.home_team)
        away_elo = team_strength(request.away_team)
        diff = home_elo - away_elo
        if not request.neutral_site:
            diff += request.home_advantage_elo

        home_goal_rate = max(0.25, 1.35 + diff / 420) * request.goal_rate_multiplier
        away_goal_rate = max(0.25, 1.15 - diff / 470) * request.goal_rate_multiplier

        home_win = draw = away_win = 0.0
        best_score = "0-0"
        best_prob = 0.0
        for home_goals in range(7):
            for away_goals in range(7):
                probability = poisson_probability(home_goal_rate, home_goals) * poisson_probability(
                    away_goal_rate, away_goals
                )
                if probability > best_prob:
                    best_prob = probability
                    best_score = f"{home_goals}-{away_goals}"
                if home_goals > away_goals:
                    home_win += probability
                elif home_goals == away_goals:
                    draw += probability
                else:
                    away_win += probability

        return ScoreDistribution(
            probabilities=normalize(
                {"home_win": home_win, "draw": draw * request.draw_bias, "away_win": away_win}
            ),
            expected_home_goals=home_goal_rate,
            expected_away_goals=away_goal_rate,
            most_likely_score=best_score,
        )


def fractional_kelly(
    probability: float,
    odds: float | None,
    fraction: float = 0.25,
    max_fraction: float = 0.05,
) -> float:
    if not odds or odds <= 1:
        return 0
    b = odds - 1
    q = 1 - probability
    full = (b * probability - q) / b
    return max(0, min(max_fraction, full * fraction))


def build_bet_signals(
    request: MatchPredictionRequest,
    probabilities: OutcomeProbability,
    odds: OddsRecord | None,
) -> list[BetSignal]:
    model_probs = probabilities.model_dump()
    odds_map = {
        "home_win": odds.win_odds if odds else None,
        "draw": odds.draw_odds if odds else None,
        "away_win": odds.lose_odds if odds else None,
    }
    market = implied_probabilities(odds) if odds else {}
    signals: list[BetSignal] = []
    for outcome, probability in model_probs.items():
        market_probability = market.get(outcome) if market else None
        edge = probability - market_probability if market_probability is not None else None
        kelly = fractional_kelly(
            probability,
            odds_map[outcome],
            fraction=request.kelly_fraction,
            max_fraction=request.max_stake_fraction,
        )
        stake = round(request.bankroll * kelly, 2)
        is_value = edge is not None and edge > request.value_edge_threshold and stake > 0
        signals.append(
            BetSignal(
                outcome=outcome,
                model_probability=round(probability, 4),
                market_probability=round(market_probability, 4)
                if market_probability is not None
                else None,
                odds=odds_map[outcome],
                edge=round(edge, 4) if edge is not None else None,
                kelly_fraction=round(kelly, 4),
                stake=stake,
                rationale=(
                    "模型概率高于去水后的市场隐含概率，满足价值投注阈值。"
                    if is_value
                    else "未达到价值投注阈值，建议观察或跳过。"
                ),
            )
        )
    return sorted(signals, key=lambda item: item.edge or -1, reverse=True)
