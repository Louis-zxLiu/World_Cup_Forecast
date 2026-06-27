from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .storage import ForecastStore


@dataclass
class FormSummary:
    """Recent-form metrics for a single national team."""

    team: str
    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def points_per_game(self) -> float:
        return self.points / self.matches if self.matches else 0.0

    @property
    def goal_diff_per_game(self) -> float:
        return (self.goals_for - self.goals_against) / self.matches if self.matches else 0.0

    def form_string(self) -> str:
        """Compact win/draw/loss record, e.g. ``5胜2平1负``."""
        return f"{self.wins}胜{self.draws}平{self.losses}负"


def summarize_recent_form(team: str, matches: list[dict]) -> FormSummary:
    """Aggregate a team's recent results from international match rows."""
    summary = FormSummary(team=team)
    for row in matches:
        is_home = row["home_team"] == team
        gf = int(row["home_score"] if is_home else row["away_score"])
        ga = int(row["away_score"] if is_home else row["home_score"])
        summary.matches += 1
        summary.goals_for += gf
        summary.goals_against += ga
        if gf > ga:
            summary.wins += 1
        elif gf == ga:
            summary.draws += 1
        else:
            summary.losses += 1
    return summary


def team_form(store: "ForecastStore", team: str, limit: int = 10) -> FormSummary:
    """Load and summarize a team's most recent international matches."""
    rows = store.get_recent_intl_for_team(team, limit=limit)
    return summarize_recent_form(team, rows)
