from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

import duckdb

from .config import get_config
from .schemas import GroupStandingRecord, LLMSettings, OddsRecord


class ForecastStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or get_config().db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.conn = duckdb.connect(str(self.db_path))
        self.init_schema()

    def init_schema(self) -> None:
        with self._lock:
            self.conn.execute("CREATE SEQUENCE IF NOT EXISTS match_results_seq START 1")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_settings (
                    id INTEGER PRIMARY KEY,
                    base_url VARCHAR NOT NULL,
                    api_key VARCHAR NOT NULL,
                    model VARCHAR NOT NULL,
                    temperature DOUBLE NOT NULL,
                    timeout_seconds INTEGER NOT NULL,
                    enabled BOOLEAN NOT NULL,
                    updated_at TIMESTAMP DEFAULT current_timestamp
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS odds_records (
                    match_id VARCHAR,
                    kickoff_time TIMESTAMP,
                    home_team VARCHAR,
                    away_team VARCHAR,
                    play_type VARCHAR,
                    handicap VARCHAR,
                    win_odds DOUBLE,
                    draw_odds DOUBLE,
                    lose_odds DOUBLE,
                    source VARCHAR,
                    source_url VARCHAR,
                    scraped_at TIMESTAMP,
                    raw_json VARCHAR
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS match_results (
                    id INTEGER PRIMARY KEY DEFAULT nextval('match_results_seq'),
                    tournament VARCHAR,
                    year INTEGER,
                    stage VARCHAR,
                    date DATE,
                    home_team VARCHAR,
                    away_team VARCHAR,
                    home_score INTEGER,
                    away_score INTEGER,
                    neutral BOOLEAN,
                    inserted_at TIMESTAMP DEFAULT current_timestamp
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS team_elo (
                    team VARCHAR PRIMARY KEY,
                    elo DOUBLE NOT NULL,
                    updated_at TIMESTAMP DEFAULT current_timestamp
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id VARCHAR PRIMARY KEY,
                    run_at TIMESTAMP DEFAULT current_timestamp,
                    model_version VARCHAR,
                    payload_json VARCHAR NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id VARCHAR PRIMARY KEY,
                    payload_json VARCHAR NOT NULL,
                    created_at TIMESTAMP DEFAULT current_timestamp
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS live_matches (
                    match_id VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    short_name VARCHAR,
                    date VARCHAR,
                    stage VARCHAR,
                    status_state VARCHAR,
                    status_name VARCHAR,
                    completed BOOLEAN,
                    display_clock VARCHAR,
                    venue VARCHAR,
                    home_team VARCHAR,
                    home_abbr VARCHAR,
                    home_score INTEGER,
                    away_team VARCHAR,
                    away_abbr VARCHAR,
                    away_score INTEGER,
                    home_odds DOUBLE,
                    away_odds DOUBLE,
                    fetched_at TIMESTAMP DEFAULT current_timestamp
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS espn_teams (
                    team_id VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    short_name VARCHAR,
                    abbreviation VARCHAR,
                    location VARCHAR,
                    color VARCHAR,
                    alternate_color VARCHAR,
                    logo_url VARCHAR,
                    slug VARCHAR,
                    fetched_at TIMESTAMP DEFAULT current_timestamp
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS group_standings (
                    group_name VARCHAR,
                    rank INTEGER,
                    team VARCHAR,
                    played INTEGER,
                    wins INTEGER,
                    draws INTEGER,
                    losses INTEGER,
                    goals_for INTEGER,
                    goals_against INTEGER,
                    goal_difference INTEGER,
                    points INTEGER,
                    source VARCHAR,
                    source_url VARCHAR,
                    scraped_at TIMESTAMP
                )
                """
            )

    def _count(self, table: str) -> int:
        with self._lock:
            row = self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0

    def match_count(self) -> int:
        return self._count("match_results")

    def clear_match_results(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM match_results")

    def insert_match_results(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        with self._lock:
            self.conn.executemany(
                """
                INSERT INTO match_results
                (tournament, year, stage, date, home_team, away_team, home_score, away_score, neutral)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    [
                        r["tournament"],
                        r["year"],
                        r["stage"],
                        r["date"],
                        r["home_team"],
                        r["away_team"],
                        r["home_score"],
                        r["away_score"],
                        r["neutral"],
                    ]
                    for r in rows
                ],
            )
        return len(rows)

    def get_match_results(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT tournament, year, stage, date, home_team, away_team,
                       home_score, away_score, neutral
                FROM match_results
                ORDER BY date
                """
            ).fetchall()
        keys = [
            "tournament",
            "year",
            "stage",
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "neutral",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def upsert_team_elo(self, elo_map: dict[str, float]) -> None:
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO team_elo (team, elo) VALUES (?, ?)",
                [[team, elo] for team, elo in elo_map.items()],
            )

    def get_team_elo(self) -> dict[str, float]:
        with self._lock:
            rows = self.conn.execute("SELECT team, elo FROM team_elo").fetchall()
        return {row[0]: row[1] for row in rows}

    def save_backtest_run(self, run_id: str, model_version: str, payload: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO backtest_runs (id, model_version, payload_json) VALUES (?, ?, ?)",
                [run_id, model_version, payload],
            )

    def list_backtest_runs(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id, run_at, model_version, payload_json
                FROM backtest_runs
                ORDER BY run_at DESC
                LIMIT 20
                """
            ).fetchall()
        return [
            {"id": row[0], "run_at": row[1], "model_version": row[2], "payload": json.loads(row[3])}
            for row in rows
        ]

    def get_llm_settings(self) -> LLMSettings:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT base_url, api_key, model, temperature, timeout_seconds, enabled
                FROM llm_settings
                WHERE id = 1
                """
            ).fetchone()
        if not row:
            return LLMSettings()
        return LLMSettings(
            base_url=row[0],
            api_key=row[1],
            model=row[2],
            temperature=row[3],
            timeout_seconds=row[4],
            enabled=row[5],
        )

    def save_llm_settings(self, settings: LLMSettings) -> LLMSettings:
        with self._lock:
            self.conn.execute("DELETE FROM llm_settings WHERE id = 1")
            self.conn.execute(
                """
                INSERT INTO llm_settings
                (id, base_url, api_key, model, temperature, timeout_seconds, enabled)
                VALUES (1, ?, ?, ?, ?, ?, ?)
                """,
                [
                    settings.base_url.rstrip("/"),
                    settings.api_key,
                    settings.model,
                    settings.temperature,
                    settings.timeout_seconds,
                    settings.enabled,
                ],
            )
        return settings

    def insert_odds(self, records: list[OddsRecord]) -> None:
        if not records:
            return
        with self._lock:
            self.conn.executemany(
                """
                INSERT INTO odds_records
                (match_id, kickoff_time, home_team, away_team, play_type, handicap,
                 win_odds, draw_odds, lose_odds, source, source_url, scraped_at, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    [
                        r.match_id,
                        r.kickoff_time,
                        r.home_team,
                        r.away_team,
                        r.play_type,
                        r.handicap,
                        r.win_odds,
                        r.draw_odds,
                        r.lose_odds,
                        r.source,
                        r.source_url,
                        r.scraped_at,
                        json.dumps(r.raw, ensure_ascii=False),
                    ]
                    for r in records
                ],
            )

    def latest_odds(self, limit: int = 100) -> list[OddsRecord]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT match_id, kickoff_time, home_team, away_team, play_type, handicap,
                       win_odds, draw_odds, lose_odds, source, source_url, scraped_at, raw_json
                FROM odds_records
                QUALIFY row_number() OVER (
                    PARTITION BY match_id, play_type, coalesce(handicap, '')
                    ORDER BY scraped_at DESC
                ) = 1
                ORDER BY scraped_at DESC, kickoff_time NULLS LAST
                LIMIT ?
                """,
                [limit],
            ).fetchall()
        return [
            OddsRecord(
                match_id=row[0],
                kickoff_time=row[1],
                home_team=row[2],
                away_team=row[3],
                play_type=row[4],
                handicap=row[5],
                win_odds=row[6],
                draw_odds=row[7],
                lose_odds=row[8],
                source=row[9],
                source_url=row[10],
                scraped_at=row[11],
                raw=json.loads(row[12] or "{}"),
            )
            for row in rows
        ]

    def odds_count(self) -> int:
        return self._count("odds_records")

    def find_match_odds(self, home_team: str, away_team: str) -> OddsRecord | None:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT match_id, kickoff_time, home_team, away_team, play_type, handicap,
                       win_odds, draw_odds, lose_odds, source, source_url, scraped_at, raw_json
                FROM odds_records
                WHERE home_team = ? AND away_team = ? AND play_type IN ('胜平负', 'spf')
                ORDER BY scraped_at DESC
                LIMIT 1
                """,
                [home_team, away_team],
            ).fetchall()
        if not rows:
            return None
        row = rows[0]
        return OddsRecord(
            match_id=row[0],
            kickoff_time=row[1],
            home_team=row[2],
            away_team=row[3],
            play_type=row[4],
            handicap=row[5],
            win_odds=row[6],
            draw_odds=row[7],
            lose_odds=row[8],
            source=row[9],
            source_url=row[10],
            scraped_at=row[11],
            raw=json.loads(row[12] or "{}"),
        )

    def upsert_live_matches(self, matches: list[dict]) -> int:
        if not matches:
            return 0
        with self._lock:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO live_matches
                (match_id, name, short_name, date, stage, status_state, status_name,
                 completed, display_clock, venue, home_team, home_abbr, home_score,
                 away_team, away_abbr, away_score, home_odds, away_odds, fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    [
                        m["match_id"],
                        m["name"],
                        m["short_name"],
                        m["date"],
                        m["stage"],
                        m["status_state"],
                        m["status_name"],
                        m["completed"],
                        m["display_clock"],
                        m["venue"],
                        m["home_team"],
                        m["home_abbr"],
                        m["home_score"],
                        m["away_team"],
                        m["away_abbr"],
                        m["away_score"],
                        m["home_odds"],
                        m["away_odds"],
                        m["fetched_at"],
                    ]
                    for m in matches
                ],
            )
        return len(matches)

    def get_live_matches(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT match_id, name, short_name, date, stage, status_state, status_name,
                       completed, display_clock, venue, home_team, home_abbr, home_score,
                       away_team, away_abbr, away_score, home_odds, away_odds, fetched_at
                FROM live_matches
                ORDER BY date ASC
                """
            ).fetchall()
        keys = [
            "match_id",
            "name",
            "short_name",
            "date",
            "stage",
            "status_state",
            "status_name",
            "completed",
            "display_clock",
            "venue",
            "home_team",
            "home_abbr",
            "home_score",
            "away_team",
            "away_abbr",
            "away_score",
            "home_odds",
            "away_odds",
            "fetched_at",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def live_match_count(self) -> int:
        return self._count("live_matches")

    def upsert_espn_teams(self, teams: list[dict]) -> int:
        if not teams:
            return 0
        with self._lock:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO espn_teams
                (team_id, name, short_name, abbreviation, location, color,
                 alternate_color, logo_url, slug)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                [
                    [
                        t["team_id"],
                        t["name"],
                        t["short_name"],
                        t["abbreviation"],
                        t["location"],
                        t["color"],
                        t["alternate_color"],
                        t["logo_url"],
                        t["slug"],
                    ]
                    for t in teams
                ],
            )
        return len(teams)

    def get_espn_teams(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT team_id, name, short_name, abbreviation, location, color,
                       alternate_color, logo_url, slug
                FROM espn_teams
                ORDER BY name
                """
            ).fetchall()
        keys = [
            "team_id",
            "name",
            "short_name",
            "abbreviation",
            "location",
            "color",
            "alternate_color",
            "logo_url",
            "slug",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def replace_group_standings(self, records: list[GroupStandingRecord]) -> int:
        if not records:
            return 0
        with self._lock:
            self.conn.execute("DELETE FROM group_standings")
            self.conn.executemany(
                """
                INSERT INTO group_standings
                (group_name, rank, team, played, wins, draws, losses, goals_for,
                 goals_against, goal_difference, points, source, source_url, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    [
                        r.group_name,
                        r.rank,
                        r.team,
                        r.played,
                        r.wins,
                        r.draws,
                        r.losses,
                        r.goals_for,
                        r.goals_against,
                        r.goal_difference,
                        r.points,
                        r.source,
                        r.source_url,
                        r.scraped_at,
                    ]
                    for r in records
                ],
            )
        return len(records)

    def get_group_standings(self) -> list[GroupStandingRecord]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT group_name, rank, team, played, wins, draws, losses, goals_for,
                       goals_against, goal_difference, points, source, source_url, scraped_at
                FROM group_standings
                ORDER BY group_name, rank
                """
            ).fetchall()
        return [
            GroupStandingRecord(
                group_name=row[0],
                rank=row[1],
                team=row[2],
                played=row[3],
                wins=row[4],
                draws=row[5],
                losses=row[6],
                goals_for=row[7],
                goals_against=row[8],
                goal_difference=row[9],
                points=row[10],
                source=row[11],
                source_url=row[12],
                scraped_at=row[13],
            )
            for row in rows
        ]

    def standing_count(self) -> int:
        return self._count("group_standings")

    def save_report(self, report_id: str, payload: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO reports (id, payload_json) VALUES (?, ?)",
                [report_id, payload],
            )

    def get_report(self, report_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT payload_json, created_at FROM reports WHERE id = ?", [report_id]
            ).fetchone()
        if not row:
            return None
        return {"id": report_id, "payload": row[0], "created_at": row[1]}
