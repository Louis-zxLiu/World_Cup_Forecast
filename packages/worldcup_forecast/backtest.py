from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from .ingest import EloUpdater, _elo_expected, _elo_update, K_FACTORS, HOME_ADVANTAGE
from .modeling import BASE_ELO, BaselineForecastModel, poisson_probability, normalize
from .schemas import BacktestMetrics, BacktestRunRequest, BacktestRunResult, MatchPredictionRequest
from .storage import ForecastStore


def _poisson_match_probs(home_rate: float, away_rate: float) -> tuple[float, float, float]:
    hw = draw = aw = 0.0
    for hg in range(7):
        for ag in range(7):
            p = (math.exp(-home_rate) * home_rate ** hg / math.factorial(hg)) * \
                (math.exp(-away_rate) * away_rate ** ag / math.factorial(ag))
            if hg > ag:
                hw += p
            elif hg == ag:
                draw += p
            else:
                aw += p
    total = hw + draw + aw
    return hw / total, draw / total, aw / total


def _brier(probs: list[tuple[float, float, float]], actuals: list[int]) -> float:
    total = 0.0
    for (ph, pd, pa), actual in zip(probs, actuals):
        oh = 1.0 if actual == 0 else 0.0
        od = 1.0 if actual == 1 else 0.0
        oa = 1.0 if actual == 2 else 0.0
        total += (ph - oh) ** 2 + (pd - od) ** 2 + (pa - oa) ** 2
    return total / len(probs) if probs else 0.0


def _log_loss(probs: list[tuple[float, float, float]], actuals: list[int]) -> float:
    eps = 1e-7
    total = 0.0
    for (ph, pd, pa), actual in zip(probs, actuals):
        p_correct = [ph, pd, pa][actual]
        total -= math.log(max(p_correct, eps))
    return total / len(probs) if probs else 0.0


def _max_drawdown(equity: list[float]) -> float:
    peak = equity[0] if equity else 0.0
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


class BacktestRunner:
    def __init__(self, store: ForecastStore | None = None) -> None:
        self.store = store or ForecastStore()

    def run(
        self,
        years: list[int] | None = None,
        params: BacktestRunRequest | None = None,
    ) -> BacktestRunResult:
        params = params or BacktestRunRequest()
        if years is not None:
            params = params.model_copy(update={"years": years})
        years = params.years

        all_results = self.store.get_match_results()
        run_id = str(uuid.uuid4())
        metrics_list: list[BacktestMetrics] = []

        for year in years:
            year_results = [r for r in all_results if int(r["year"]) == year]
            if not year_results:
                continue

            # Build rolling Elo from all matches BEFORE this tournament
            prior_results = [r for r in all_results if int(r["year"]) < year]
            elo: dict[str, float] = dict(BASE_ELO)
            for row in prior_results:
                home, away = row["home_team"], row["away_team"]
                elo.setdefault(home, 1500.0)
                elo.setdefault(away, 1500.0)
                k = K_FACTORS.get(row["stage"], 40)
                elo[home], elo[away] = _elo_update(
                    elo[home], elo[away],
                    int(row["home_score"]), int(row["away_score"]),
                    k, bool(row["neutral"]),
                )

            model_probs: list[tuple[float, float, float]] = []
            baseline_probs: list[tuple[float, float, float]] = []
            actuals: list[int] = []
            bankroll = params.initial_bankroll
            equity = [bankroll]
            baseline_model = BaselineForecastModel()

            for row in year_results:
                home, away = row["home_team"], row["away_team"]
                hs, as_ = int(row["home_score"]), int(row["away_score"])
                if hs > as_:
                    actual = 0
                elif hs == as_:
                    actual = 1
                else:
                    actual = 2
                actuals.append(actual)

                h_elo = elo.get(home, BASE_ELO.get(home, 1500.0))
                a_elo = elo.get(away, BASE_ELO.get(away, 1500.0))
                diff = h_elo - a_elo + (0 if bool(row["neutral"]) else HOME_ADVANTAGE)
                h_rate = max(0.25, 1.35 + diff / 420)
                a_rate = max(0.25, 1.15 - diff / 470)
                ph, pd, pa = _poisson_match_probs(h_rate, a_rate)
                model_probs.append((ph, pd, pa))

                # baseline: same but using BASE_ELO only
                req = MatchPredictionRequest(
                    home_team=home, away_team=away, neutral_site=bool(row["neutral"])
                )
                bd = baseline_model.predict_score_distribution(req)
                baseline_probs.append((bd.probabilities.home_win, bd.probabilities.draw, bd.probabilities.away_win))

                # Research betting model: use configurable flat market odds and fractional Kelly.
                best_idx = max(range(3), key=lambda i: [ph, pd, pa][i])
                flat_odds = [params.home_odds, params.draw_odds, params.away_odds][best_idx]
                market_implied = 1 / flat_odds
                model_p = [ph, pd, pa][best_idx]
                edge = model_p - market_implied
                if edge > params.edge_threshold and bankroll > 0:
                    b = flat_odds - 1
                    full_kelly = max(0.0, (b * model_p - (1 - model_p)) / b)
                    stake_fraction = min(params.max_stake_fraction, full_kelly * params.kelly_fraction)
                    stake = min(bankroll * stake_fraction, bankroll)
                    if actual == best_idx:
                        bankroll += stake * (flat_odds - 1)
                    else:
                        bankroll -= stake
                equity.append(bankroll)

                # Update elo after match
                elo.setdefault(home, 1500.0)
                elo.setdefault(away, 1500.0)
                k = K_FACTORS.get(row["stage"], 40)
                elo[home], elo[away] = _elo_update(
                    elo[home], elo[away], hs, as_, k, bool(row["neutral"])
                )

            if not actuals:
                continue

            roi = (equity[-1] - params.initial_bankroll) / params.initial_bankroll
            metrics_list.append(
                BacktestMetrics(
                    tournament_year=year,
                    brier=round(_brier(model_probs, actuals), 4),
                    log_loss=round(_log_loss(model_probs, actuals), 4),
                    roi=round(roi, 4),
                    max_drawdown=round(_max_drawdown(equity), 4),
                    record_count=len(actuals),
                    baseline_brier=round(_brier(baseline_probs, actuals), 4),
                    baseline_log_loss=round(_log_loss(baseline_probs, actuals), 4),
                )
            )

        result = BacktestRunResult(
            id=run_id,
            run_at=__import__("datetime").datetime.utcnow(),
            metrics=metrics_list,
            params=params,
        )
        self.store.save_backtest_run(
            run_id,
            "elo-poisson-v1",
            json.dumps(result.model_dump(), default=str),
        )
        return result
