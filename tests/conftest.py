from __future__ import annotations

import os
import tempfile

import duckdb
import pytest
from pathlib import Path

from worldcup_forecast.storage import ForecastStore

_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="worldcup_forecast_tests_"))
os.environ["WORLD_CUP_DB_PATH"] = str(_TEST_DATA_DIR / "test_api.duckdb")
os.environ["ODDS_SNAPSHOT_DIR"] = str(_TEST_DATA_DIR / "odds_snapshots")


@pytest.fixture
def tmp_store(tmp_path: Path) -> ForecastStore:
    db = tmp_path / "test.duckdb"
    return ForecastStore(db_path=db)
