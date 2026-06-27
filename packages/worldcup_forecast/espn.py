from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx

from .schemas import GroupStandingRecord

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/teams"
STANDINGS_URLS = [
    "https://site.web.api.espn.com/apis/v2/sports/soccer/fifa.world/standings",
    "https://sports.core.api.espn.com/v2/sports/soccer/leagues/fifa.world/standings?lang=zh&region=cn",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ESPNMatch:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        comp = data.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0] if competitors else {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1] if len(competitors) > 1 else {})

        self.match_id: str = str(data.get("id", ""))
        self.name: str = data.get("name", "")
        self.short_name: str = data.get("shortName", "")
        self.date: str = data.get("date", "")
        self.stage: str = data.get("season", {}).get("slug", "")
        self.status_state: str = comp.get("status", {}).get("type", {}).get("state", "pre")
        self.status_name: str = comp.get("status", {}).get("type", {}).get("name", "Scheduled")
        self.completed: bool = comp.get("status", {}).get("type", {}).get("completed", False)
        self.display_clock: str = comp.get("status", {}).get("displayClock", "")
        self.venue: str = comp.get("venue", {}).get("fullName", "")

        self.home_team: str = home.get("team", {}).get("displayName", "")
        self.home_abbr: str = home.get("team", {}).get("abbreviation", "")
        self.home_score: int | None = int(home["score"]) if home.get("score") is not None else None

        self.away_team: str = away.get("team", {}).get("displayName", "")
        self.away_abbr: str = away.get("team", {}).get("abbreviation", "")
        self.away_score: int | None = int(away["score"]) if away.get("score") is not None else None

        # Odds from ESPN (DraftKings embedded)
        bet = comp.get("betDetails", {})
        self.home_odds: float | None = _safe_float(bet.get("homeTeamOdds", {}).get("moneyLine"))
        self.away_odds: float | None = _safe_float(bet.get("awayTeamOdds", {}).get("moneyLine"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "name": self.name,
            "short_name": self.short_name,
            "date": self.date,
            "stage": self.stage,
            "status_state": self.status_state,
            "status_name": self.status_name,
            "completed": self.completed,
            "display_clock": self.display_clock,
            "venue": self.venue,
            "home_team": self.home_team,
            "home_abbr": self.home_abbr,
            "home_score": self.home_score,
            "away_team": self.away_team,
            "away_abbr": self.away_abbr,
            "away_score": self.away_score,
            "home_odds": self.home_odds,
            "away_odds": self.away_odds,
            "fetched_at": _utcnow().isoformat(),
        }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ESPNProvider:
    async def fetch_scoreboard(self, date: str | None = None) -> list[ESPNMatch]:
        """Fetch scoreboard for a given date (YYYYMMDD) or today if omitted."""
        url = SCOREBOARD_URL
        if date:
            url = f"{SCOREBOARD_URL}?dates={date}"
        async with httpx.AsyncClient(timeout=15, headers=HEADERS, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        events = data.get("events", [])
        return [ESPNMatch(e) for e in events]

    async def fetch_all_matches(self) -> list[ESPNMatch]:
        """Fetch today's matches; if all are finished, also fetch tomorrow's.

        ESPN's default scoreboard returns today's events. When every match has
        completed we automatically pull the next calendar day so the UI always
        shows upcoming fixtures rather than a stale completed list.
        """
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        matches = await self.fetch_scoreboard(date=today_str)

        all_done = bool(matches) and all(m.completed for m in matches)
        if all_done:
            from datetime import timedelta
            tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
            tomorrow_str = tomorrow.strftime("%Y%m%d")
            next_matches = await self.fetch_scoreboard(date=tomorrow_str)
            # Merge: keep today's results plus tomorrow's upcoming fixtures.
            seen = {m.match_id for m in matches}
            matches = matches + [m for m in next_matches if m.match_id not in seen]

        return matches

    async def fetch_teams(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=15, headers=HEADERS, follow_redirects=True) as client:
            resp = await client.get(TEAMS_URL)
            resp.raise_for_status()
            data = resp.json()
        teams = []
        for sport in data.get("sports", []):
            for league in sport.get("leagues", []):
                for wrapper in league.get("teams", []):
                    t = wrapper.get("team", {})
                    logo = next(
                        (lg["href"] for lg in t.get("logos", []) if "default" in lg.get("rel", [])),
                        t.get("logos", [{}])[0].get("href", "") if t.get("logos") else "",
                    )
                    teams.append({
                        "team_id": str(t.get("id", "")),
                        "name": t.get("displayName", ""),
                        "short_name": t.get("shortDisplayName", ""),
                        "abbreviation": t.get("abbreviation", ""),
                        "location": t.get("location", ""),
                        "color": t.get("color", ""),
                        "alternate_color": t.get("alternateColor", ""),
                        "logo_url": logo,
                        "slug": t.get("slug", ""),
                    })
        return teams

    async def fetch_standings(self) -> list[GroupStandingRecord]:
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=15, headers=HEADERS, follow_redirects=True) as client:
            for url in STANDINGS_URLS:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    records = _parse_standings(resp.json(), url)
                    if records:
                        return records
                except Exception as exc:
                    last_error = exc
        if last_error:
            raise last_error
        return []


def _stat_value(stats: list[dict[str, Any]], *names: str) -> int:
    aliases = {name.lower() for name in names}
    for stat in stats:
        keys = [
            str(stat.get("name", "")).lower(),
            str(stat.get("abbreviation", "")).lower(),
            str(stat.get("shortDisplayName", "")).lower(),
            str(stat.get("displayName", "")).lower(),
        ]
        if any(key in aliases for key in keys):
            try:
                return int(float(stat.get("value", stat.get("displayValue", 0))))
            except (TypeError, ValueError):
                return 0
    return 0


def _parse_standings(data: dict[str, Any], source_url: str) -> list[GroupStandingRecord]:
    groups = data.get("children") or data.get("standings") or data.get("groups") or []
    if isinstance(groups, dict):
        groups = groups.get("entries") or groups.get("children") or []

    records: list[GroupStandingRecord] = []
    scraped_at = _utcnow()
    for group_index, group in enumerate(groups):
        group_name = (
            group.get("name")
            or group.get("displayName")
            or group.get("abbreviation")
            or f"Group {chr(65 + group_index)}"
        )
        entries = group.get("standings", {}).get("entries") or group.get("entries") or []
        if isinstance(entries, dict):
            entries = entries.get("entries", [])
        for index, entry in enumerate(entries):
            team_data = entry.get("team", {})
            team = (
                team_data.get("displayName")
                or team_data.get("name")
                or entry.get("displayName")
                or entry.get("name")
                or ""
            )
            if isinstance(team, dict):
                team = team.get("displayName") or team.get("name") or ""
            if not team:
                continue

            stats = entry.get("stats", [])
            goals_for = _stat_value(stats, "pointsfor", "goalsfor", "gf", "f")
            goals_against = _stat_value(stats, "pointsagainst", "goalsagainst", "ga", "a")
            records.append(
                GroupStandingRecord(
                    group_name=str(group_name),
                    rank=int(entry.get("rank") or index + 1),
                    team=str(team),
                    played=_stat_value(stats, "gamesplayed", "gp", "played", "matchesplayed"),
                    wins=_stat_value(stats, "wins", "w"),
                    draws=_stat_value(stats, "ties", "draws", "d"),
                    losses=_stat_value(stats, "losses", "l"),
                    goals_for=goals_for,
                    goals_against=goals_against,
                    goal_difference=_stat_value(
                        stats, "pointdifferential", "goaldifference", "gd", "+/-"
                    )
                    or goals_for - goals_against,
                    points=_stat_value(stats, "points", "pts"),
                    source="espn",
                    source_url=source_url,
                    scraped_at=scraped_at,
                )
            )
    return records
