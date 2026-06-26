from __future__ import annotations

import math
import pytest
from pathlib import Path

from worldcup_forecast.backtest import BacktestRunner, _brier, _log_loss, _max_drawdown
from worldcup_forecast.storage import ForecastStore
from worldcup_forecast.ingest import HistoricalIngestor, EloUpdater


def test_brier_perfect_prediction():
    probs = [(1.0, 0.0, 0.0)]
    actuals = [0]
    assert _brier(probs, actuals) == 0.0


def test_brier_worst_prediction():
    probs = [(0.0, 0.0, 1.0)]
    actuals = [0]
    assert abs(_brier(probs, actuals) - 2.0) < 1e-9


def test_brier_uniform():
    probs = [(1/3, 1/3, 1/3)]
    actuals = [0]
    b = _brier(probs, actuals)
    assert abs(b - (4/9 + 1/9 + 1/9)) < 1e-6


def test_log_loss_perfect():
    probs = [(1.0, 0.0, 0.0)]
    actuals = [0]
    assert _log_loss(probs, actuals) < 1e-5


def test_log_loss_bounded():
    probs = [(1/3, 1/3, 1/3)] * 10
    actuals = [0] * 10
    ll = _log_loss(probs, actuals)
    assert ll < math.log(3) + 0.01  # should be close to log(3)


def test_max_drawdown_no_drawdown():
    equity = [100, 110, 120, 130]
    assert _max_drawdown(equity) == 0.0


def test_max_drawdown_full_loss():
    equity = [100, 50]
    assert abs(_max_drawdown(equity) - 0.5) < 1e-9


def test_max_drawdown_recovery():
    equity = [100, 80, 90, 70, 110]
    dd = _max_drawdown(equity)
    assert abs(dd - 0.3) < 1e-6  # peak 100, trough 70 → 30%


def test_backtest_run_with_data(tmp_store: ForecastStore):
    csv_path = Path("data/wc_results.csv")
    if not csv_path.exists():
        pytest.skip("data/wc_results.csv not found")
    HistoricalIngestor(tmp_store).load_csv(csv_path)
    EloUpdater(tmp_store).update_from_results()
    runner = BacktestRunner(tmp_store)
    result = runner.run(years=[2022])
    assert len(result.metrics) == 1
    m = result.metrics[0]
    assert m.tournament_year == 2022
    assert m.record_count > 0
    assert 0.0 <= m.brier <= 2.0
    assert m.log_loss >= 0.0
    assert -1.0 <= m.roi <= 10.0
    assert 0.0 <= m.max_drawdown <= 1.0


def test_backtest_three_years(tmp_store: ForecastStore):
    csv_path = Path("data/wc_results.csv")
    if not csv_path.exists():
        pytest.skip("data/wc_results.csv not found")
    HistoricalIngestor(tmp_store).load_csv(csv_path)
    EloUpdater(tmp_store).update_from_results()
    result = BacktestRunner(tmp_store).run()
    assert len(result.metrics) == 3
    years = {m.tournament_year for m in result.metrics}
    assert years == {2014, 2018, 2022}
