from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from worldcup_forecast.ingest import (
    EloUpdater,
    HistoricalIngestor,
    _elo_update,
    _elo_expected,
    _normalize_worldcup_csv,
)
from worldcup_forecast.storage import ForecastStore


def test_elo_expected_equal_teams():
    exp = _elo_expected(1500, 1500)
    assert abs(exp - 0.5) < 1e-6


def test_elo_expected_stronger_home():
    exp = _elo_expected(1600, 1400)
    assert exp > 0.5


def test_elo_update_winner_gains_elo():
    home_new, away_new = _elo_update(1500, 1500, 2, 0, k=40, neutral=True)
    assert home_new > 1500
    assert away_new < 1500
    assert abs((home_new - 1500) + (away_new - 1500)) < 1e-6


def test_elo_update_draw_equal_teams_no_change():
    home_new, away_new = _elo_update(1500, 1500, 1, 1, k=40, neutral=True)
    assert abs(home_new - 1500) < 1e-6
    assert abs(away_new - 1500) < 1e-6


def test_load_csv_inserts_records(tmp_store: ForecastStore):
    csv_path = Path("data/wc_results.csv")
    if not csv_path.exists():
        pytest.skip("data/wc_results.csv not found")
    ingestor = HistoricalIngestor(tmp_store)
    count = ingestor.load_csv(csv_path)
    assert count > 0
    assert tmp_store.match_count() == count


def test_elo_updater_produces_rankings(tmp_store: ForecastStore):
    csv_path = Path("data/wc_results.csv")
    if not csv_path.exists():
        pytest.skip("data/wc_results.csv not found")
    HistoricalIngestor(tmp_store).load_csv(csv_path)
    elo = EloUpdater(tmp_store).update_from_results()
    assert len(elo) > 10
    assert "Brazil" in elo
    assert "Germany" in elo
    # Top teams should have elo above average
    avg = sum(elo.values()) / len(elo)
    assert elo["Brazil"] > avg or elo["Germany"] > avg


def test_load_csv_idempotent(tmp_store: ForecastStore):
    csv_path = Path("data/wc_results.csv")
    if not csv_path.exists():
        pytest.skip("data/wc_results.csv not found")
    ingestor = HistoricalIngestor(tmp_store)
    count1 = ingestor.load_csv(csv_path)
    count2 = ingestor.load_csv(csv_path)
    assert tmp_store.match_count() == count2  # second load replaces


def test_normalize_worldcup_source_csv():
    text = """tournament_name,stage_name,group_name,match_date,home_team_name,away_team_name,home_team_score,away_team_score,country_name
FIFA World Cup,group stage,Group A,2022-11-20,Qatar,Ecuador,0,2,Qatar
FIFA World Cup,final,,2022-12-18,Argentina,France,3,3,Qatar
"""
    rows = _normalize_worldcup_csv(text)

    assert len(rows) == 2
    assert rows[0]["stage"] == "Group A"
    assert rows[0]["neutral"] == "0"
    assert rows[1]["stage"] == "final"
    assert rows[1]["home_team"] == "Argentina"
