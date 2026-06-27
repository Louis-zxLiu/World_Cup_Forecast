from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from .modeling import BASE_ELO, NEUTRAL_ELO
from .schemas import WorldCupDataImportResult
from .storage import ForecastStore

DEFAULT_CSV = Path("data/wc_results.csv")
DEFAULT_INTL_CSV = Path("data/intl_results.csv")
RAW_SOURCE_DIR = Path("data/raw/worldcup_sources")

WORLD_CUP_MATCH_SOURCE_URLS = [
    "https://cdn.jsdelivr.net/gh/jfjelstul/worldcup@master/data-csv/matches.csv",
    "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv",
]

# Full archive of international men's A-team matches since 1872 (~49k rows).
# Used to build realistic Elo ratings and recent form for *every* national team,
# not just the dozen that appear frequently at World Cups.
INTERNATIONAL_RESULTS_SOURCE_URLS = [
    "https://cdn.jsdelivr.net/gh/martj42/international_results@master/results.csv",
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
]

# Substrings that identify a senior men's World Cup *finals* match (not qualifiers).
WORLD_CUP_TOURNAMENT_MARKERS = ("fifa world cup", "men's world cup")


def is_world_cup_finals(tournament: str) -> bool:
    """True for World Cup finals matches, excluding qualification rounds."""
    name = (tournament or "").lower()
    if "qualif" in name:
        return False
    return any(marker in name for marker in WORLD_CUP_TOURNAMENT_MARKERS)


K_FACTORS: dict[str, int] = {
    "Final": 60,
    "Third place": 60,
    "Semi-final": 60,
    "Quarter-final": 60,
    "Round of 16": 60,
    "Group A": 50,
    "Group B": 50,
    "Group C": 50,
    "Group D": 50,
    "Group E": 50,
    "Group F": 50,
    "Group G": 50,
    "Group H": 50,
}
HOME_ADVANTAGE = 45


# Tournament-importance weight (World Football Elo style). Friendlies move
# ratings little; World Cup finals move them a lot. Matched by case-insensitive
# substring against the ``tournament`` field of international results.
TOURNAMENT_IMPORTANCE: list[tuple[str, int]] = [
    ("fifa world cup", 60),
    ("men's world cup", 60),
    ("uefa euro", 50),
    ("copa am", 50),  # Copa América
    ("african cup of nations", 50),
    ("afc asian cup", 50),
    ("confederations cup", 45),
    ("nations league", 45),
    ("gold cup", 40),
    ("qualif", 40),  # any qualification campaign
]
DEFAULT_IMPORTANCE = 30  # friendlies and minor tournaments


def importance_for(tournament: str) -> int:
    """Map a tournament name to its Elo K-factor (importance weight)."""
    name = (tournament or "").lower()
    # Qualification matches rank below their parent competition.
    if "qualif" in name:
        return 40
    for marker, weight in TOURNAMENT_IMPORTANCE:
        if marker in name:
            return weight
    return DEFAULT_IMPORTANCE


def _goal_difference_multiplier(home_score: int, away_score: int) -> float:
    """World Football Elo goal-difference weighting.

    A one-goal win counts fully; larger margins amplify the rating change with
    diminishing returns, so blowouts matter more than narrow wins but not
    without bound.
    """
    margin = abs(home_score - away_score)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8.0



def _pick(raw: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = raw.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def _parse_int(value: str) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _normalize_stage(raw: dict[str, str]) -> str:
    group_name = _pick(raw, "group_name", "group", "Group")
    stage_name = _pick(raw, "stage_name", "stage", "Stage")
    if group_name and group_name.lower() not in {"not applicable", "na", "n/a", "none"}:
        return group_name
    if stage_name.lower() in {"group stage", "first group stage", "second group stage"} and group_name:
        return group_name
    return stage_name or "Unknown"


def _normalize_worldcup_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(text.splitlines())
    rows: list[dict[str, Any]] = []
    for raw in reader:
        date_text = _pick(raw, "date", "match_date", "Match Date")
        home = _pick(raw, "home_team", "home_team_name", "home", "Home Team")
        away = _pick(raw, "away_team", "away_team_name", "away", "Away Team")
        home_score = _parse_int(_pick(raw, "home_score", "home_team_score", "Home Goals"))
        away_score = _parse_int(_pick(raw, "away_score", "away_team_score", "Away Goals"))
        if not (date_text and home and away) or home_score is None or away_score is None:
            continue

        year_text = _pick(raw, "year", "tournament_year")
        year = _parse_int(year_text) or int(date_text[:4])
        tournament = _pick(raw, "tournament", "tournament_name") or "FIFA World Cup"
        country = _pick(raw, "country_name", "host_country")
        neutral = "0" if country and country.lower() == home.lower() else "1"
        rows.append(
            {
                "tournament": tournament,
                "year": year,
                "stage": _normalize_stage(raw),
                "date": date_text[:10],
                "home_team": home,
                "home_score": home_score,
                "away_team": away,
                "away_score": away_score,
                "neutral": neutral,
            }
        )
    rows.sort(key=lambda row: (row["year"], row["date"], row["stage"], row["home_team"]))
    return rows


def _write_standard_csv(rows: list[dict[str, Any]], path: Path = DEFAULT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "tournament",
                "year",
                "stage",
                "date",
                "home_team",
                "home_score",
                "away_team",
                "away_score",
                "neutral",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


class WorldCupDataDownloader:
    def __init__(
        self,
        source_urls: list[str] | None = None,
        output_path: Path = DEFAULT_CSV,
        raw_dir: Path = RAW_SOURCE_DIR,
    ) -> None:
        self.source_urls = source_urls or WORLD_CUP_MATCH_SOURCE_URLS
        self.output_path = output_path
        self.raw_dir = raw_dir

    def download_and_prepare(self, force: bool = True) -> WorldCupDataImportResult:
        if not force and self.output_path.exists():
            rows = _normalize_worldcup_csv(self.output_path.read_text(encoding="utf-8"))
            return self._result("cached", "", "", rows, "使用本地已整理数据。")

        errors: list[str] = []
        for url in self.source_urls:
            try:
                text = self._fetch(url)
                rows = _normalize_worldcup_csv(text)
                if not rows:
                    raise ValueError("下载成功，但没有解析出比赛记录。")
                raw_path = self._save_raw(text, url)
                _write_standard_csv(rows, self.output_path)
                return self._result("downloaded", url, str(raw_path), rows, "联网下载并整理完成。")
            except Exception as exc:
                errors.append(f"{url}: {exc}")

        if self.output_path.exists():
            rows = _normalize_worldcup_csv(self.output_path.read_text(encoding="utf-8"))
            return self._result(
                "fallback",
                "",
                "",
                rows,
                "网络下载失败，已使用本地整理数据。原因：" + " | ".join(errors),
            )
        raise RuntimeError("历史世界杯数据下载失败，且本地没有可用 CSV：" + " | ".join(errors))

    def _fetch(self, url: str) -> str:
        headers = {"User-Agent": "WorldCupForecast/0.1 (+local research)"}
        with httpx.Client(timeout=20, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text

    def _save_raw(self, text: str, url: str) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        name = "fjelstul_matches.csv" if "fjelstul" in url else "worldcup_matches.csv"
        path = self.raw_dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def _result(
        self,
        status: str,
        source_url: str,
        raw_path: str,
        rows: list[dict[str, Any]],
        message: str,
    ) -> WorldCupDataImportResult:
        years = sorted({int(row["year"]) for row in rows})
        return WorldCupDataImportResult(
            status=status,
            source_url=source_url,
            raw_path=raw_path,
            output_path=str(self.output_path),
            rows=len(rows),
            years=years,
            message=message,
        )


def _elo_expected(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def _elo_update(
    elo_home: float,
    elo_away: float,
    home_score: int,
    away_score: int,
    k: int,
    neutral: bool,
    use_goal_difference: bool = False,
) -> tuple[float, float]:
    adj_home = elo_home + (0 if neutral else HOME_ADVANTAGE)
    exp_home = _elo_expected(adj_home, elo_away)
    if home_score > away_score:
        result = 1.0
    elif home_score == away_score:
        result = 0.5
    else:
        result = 0.0
    effective_k = k
    if use_goal_difference:
        effective_k = k * _goal_difference_multiplier(home_score, away_score)
    delta = effective_k * (result - exp_home)
    return elo_home + delta, elo_away - delta


class HistoricalIngestor:
    def __init__(self, store: ForecastStore | None = None) -> None:
        self.store = store or ForecastStore()

    def load_csv(self, path: Path = DEFAULT_CSV) -> int:
        rows: list[dict[str, Any]] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                rows.append(
                    {
                        "tournament": raw["tournament"].strip(),
                        "year": int(raw["year"]),
                        "stage": raw["stage"].strip(),
                        "date": date.fromisoformat(raw["date"].strip()),
                        "home_team": raw["home_team"].strip(),
                        "away_team": raw["away_team"].strip(),
                        "home_score": int(raw["home_score"]),
                        "away_score": int(raw["away_score"]),
                        "neutral": raw["neutral"].strip() in ("1", "true", "True"),
                    }
                )
        # Clear existing and reload to keep idempotent
        self.store.clear_match_results()
        return self.store.insert_match_results(rows)


class InternationalResultsDownloader:
    """Download the full martj42 archive of international men's matches."""

    def __init__(
        self,
        source_urls: list[str] | None = None,
        output_path: Path = DEFAULT_INTL_CSV,
        raw_dir: Path = RAW_SOURCE_DIR,
    ) -> None:
        self.source_urls = source_urls or INTERNATIONAL_RESULTS_SOURCE_URLS
        self.output_path = output_path
        self.raw_dir = raw_dir

    def download(self, force: bool = True) -> Path:
        if not force and self.output_path.exists():
            return self.output_path
        errors: list[str] = []
        for url in self.source_urls:
            try:
                text = self._fetch(url)
                if "home_team" not in text.splitlines()[0]:
                    raise ValueError("unexpected header")
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_path.write_text(text, encoding="utf-8")
                return self.output_path
            except Exception as exc:  # noqa: BLE001 - try next mirror
                errors.append(f"{url}: {exc}")
        if self.output_path.exists():
            return self.output_path
        raise RuntimeError("国际比赛数据下载失败，且本地无缓存：" + " | ".join(errors))

    def _fetch(self, url: str) -> str:
        headers = {"User-Agent": "WorldCupForecast/0.1 (+local research)"}
        with httpx.Client(timeout=60, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text


class InternationalResultsIngestor:
    """Load the international archive into the dedicated ``intl_results`` table."""

    def __init__(self, store: ForecastStore | None = None) -> None:
        self.store = store or ForecastStore()

    def load_csv(self, path: Path = DEFAULT_INTL_CSV) -> int:
        rows: list[dict[str, Any]] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                home_score = _parse_int(raw.get("home_score", ""))
                away_score = _parse_int(raw.get("away_score", ""))
                date_text = (raw.get("date") or "").strip()
                home = (raw.get("home_team") or "").strip()
                away = (raw.get("away_team") or "").strip()
                if home_score is None or away_score is None or not (date_text and home and away):
                    continue
                try:
                    match_date = date.fromisoformat(date_text[:10])
                except ValueError:
                    continue
                neutral_raw = (raw.get("neutral") or "").strip().lower()
                rows.append(
                    {
                        "date": match_date,
                        "home_team": home,
                        "away_team": away,
                        "home_score": home_score,
                        "away_score": away_score,
                        "tournament": (raw.get("tournament") or "Friendly").strip(),
                        "neutral": neutral_raw in ("1", "true", "t", "yes"),
                    }
                )
        rows.sort(key=lambda r: r["date"])
        self.store.clear_intl_results()
        return self.store.insert_intl_results(rows)


class EloUpdater:
    def __init__(self, store: ForecastStore | None = None) -> None:
        self.store = store or ForecastStore()

    def update_from_results(self) -> dict[str, float]:
        """Build Elo ratings, preferring the full international archive.

        When ``intl_results`` is populated we use the entire match history with
        tournament-importance and goal-difference weighting. Otherwise we fall
        back to the World Cup-only ``match_results`` table.
        """
        intl_rows = self.store.get_intl_results()
        if intl_rows:
            return self._update_from_intl(intl_rows)
        return self._update_from_world_cup()

    def _update_from_intl(self, rows: list[dict]) -> dict[str, float]:
        elo: dict[str, float] = dict(BASE_ELO)
        for row in rows:
            home, away = row["home_team"], row["away_team"]
            elo.setdefault(home, NEUTRAL_ELO)
            elo.setdefault(away, NEUTRAL_ELO)
            k = importance_for(row["tournament"])
            elo[home], elo[away] = _elo_update(
                elo[home], elo[away],
                int(row["home_score"]), int(row["away_score"]),
                k, bool(row["neutral"]),
                use_goal_difference=True,
            )
        self.store.upsert_team_elo(elo)
        return elo

    def _update_from_world_cup(self) -> dict[str, float]:
        rows = self.store.get_match_results()
        elo: dict[str, float] = dict(BASE_ELO)
        for row in rows:
            home, away = row["home_team"], row["away_team"]
            elo.setdefault(home, NEUTRAL_ELO)
            elo.setdefault(away, NEUTRAL_ELO)
            k = K_FACTORS.get(row["stage"], 40)
            new_home, new_away = _elo_update(
                elo[home], elo[away],
                int(row["home_score"]), int(row["away_score"]),
                k, bool(row["neutral"]),
            )
            elo[home] = new_home
            elo[away] = new_away
        self.store.upsert_team_elo(elo)
        return elo
