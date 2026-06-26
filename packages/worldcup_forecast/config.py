from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class AppConfig(BaseModel):
    db_path: Path = Field(default=Path("data/worldcup.duckdb"))
    odds_snapshot_dir: Path = Field(default=Path("data/raw/odds_snapshots"))
    china_lottery_odds_url: str = Field(default="https://trade.500.com/jczq/")
    fallback_sporttery_url: str = Field(default="https://www.sporttery.cn/jc/jczq/")


@lru_cache
def get_config() -> AppConfig:
    return AppConfig(
        db_path=Path(os.getenv("WORLD_CUP_DB_PATH", "data/worldcup.duckdb")),
        odds_snapshot_dir=Path(os.getenv("ODDS_SNAPSHOT_DIR", "data/raw/odds_snapshots")),
        china_lottery_odds_url=os.getenv(
            "CHINA_LOTTERY_ODDS_URL", "https://trade.500.com/jczq/"
        ),
        fallback_sporttery_url=os.getenv(
            "SPORTTERY_ODDS_URL", "https://www.sporttery.cn/jc/jczq/"
        ),
    )

