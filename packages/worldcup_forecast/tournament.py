from __future__ import annotations

import math
import random
from typing import Any

from .modeling import BASE_ELO, poisson_probability
from .schemas import (
    GroupStandingRecord,
    TournamentBracket,
    TournamentFixture,
    TournamentTeamProbability,
)


def _team_elo(team: str, elo_map: dict[str, float]) -> float:
    return elo_map.get(team, BASE_ELO.get(team, 1500.0))


def _goal_rates(elo_home: float, elo_away: float, neutral: bool = True) -> tuple[float, float]:
    diff = elo_home - elo_away + (0 if neutral else 45)
    home_rate = max(0.25, 1.35 + diff / 420)
    away_rate = max(0.25, 1.15 - diff / 470)
    return home_rate, away_rate


def _simulate_match(
    home: str, away: str, elo_map: dict[str, float], neutral: bool = True
) -> tuple[int, int]:
    h_rate, a_rate = _goal_rates(_team_elo(home, elo_map), _team_elo(away, elo_map), neutral)
    home_goals = _poisson_sample(h_rate)
    away_goals = _poisson_sample(a_rate)
    return home_goals, away_goals


def _poisson_sample(lam: float) -> int:
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1


def _penalty_winner(home: str, away: str) -> str:
    return random.choice([home, away])


def _empty_table(teams: list[str]) -> dict[str, dict[str, int]]:
    return {
        team: {"points": 0, "gd": 0, "gf": 0, "played": 0}
        for team in teams
        if not team.startswith("__bye")
    }


def _apply_group_result(table: dict[str, dict[str, int]], home: str, away: str, hg: int, ag: int) -> None:
    if home not in table or away not in table:
        return
    table[home]["played"] += 1
    table[away]["played"] += 1
    table[home]["gf"] += hg
    table[away]["gf"] += ag
    table[home]["gd"] += hg - ag
    table[away]["gd"] += ag - hg
    if hg > ag:
        table[home]["points"] += 3
    elif hg == ag:
        table[home]["points"] += 1
        table[away]["points"] += 1
    else:
        table[away]["points"] += 3


def _rank_group(table: dict[str, dict[str, int]]) -> list[str]:
    return sorted(
        table,
        key=lambda t: (table[t]["points"], table[t]["gd"], table[t]["gf"], _team_elo(t, {})),
        reverse=True,
    )


def _simulate_group(teams: list[str], elo_map: dict[str, float]) -> list[str]:
    table = _empty_table(teams)
    for i, home in enumerate(teams):
        for away in teams[i + 1:]:
            hg, ag = _simulate_match(home, away, elo_map, neutral=True)
            _apply_group_result(table, home, away, hg, ag)
    ranked = _rank_group(table)
    return ranked[:2]


def _standings_to_groups(
    standings: list[GroupStandingRecord],
) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, str]]:
    groups: dict[str, dict[str, dict[str, int]]] = {}
    team_to_group: dict[str, str] = {}
    for row in standings:
        group = row.group_name or "Group"
        groups.setdefault(group, {})
        groups[group][row.team] = {
            "points": row.points,
            "gd": row.goal_difference,
            "gf": row.goals_for,
            "played": row.played,
        }
        team_to_group[row.team] = group
    return groups, team_to_group


def _fixtures_by_group(
    fixtures: list[TournamentFixture],
    team_to_group: dict[str, str],
) -> dict[str, list[TournamentFixture]]:
    grouped: dict[str, list[TournamentFixture]] = {}
    for fixture in fixtures:
        group = fixture.group_name
        if not group:
            home_group = team_to_group.get(fixture.home_team)
            away_group = team_to_group.get(fixture.away_team)
            group = home_group if home_group and home_group == away_group else None
        if not group:
            continue
        grouped.setdefault(group, []).append(fixture)
    return grouped


def _simulate_group_from_state(
    table: dict[str, dict[str, int]],
    fixtures: list[TournamentFixture],
    elo_map: dict[str, float],
    standings_are_current: bool,
) -> list[str]:
    simulated = {team: dict(values) for team, values in table.items()}
    for fixture in fixtures:
        if fixture.home_team not in simulated:
            simulated[fixture.home_team] = {"points": 0, "gd": 0, "gf": 0, "played": 0}
        if fixture.away_team not in simulated:
            simulated[fixture.away_team] = {"points": 0, "gd": 0, "gf": 0, "played": 0}

        if fixture.completed:
            if not standings_are_current and fixture.home_score is not None and fixture.away_score is not None:
                _apply_group_result(
                    simulated,
                    fixture.home_team,
                    fixture.away_team,
                    fixture.home_score,
                    fixture.away_score,
                )
            continue

        hg, ag = _simulate_match(fixture.home_team, fixture.away_team, elo_map, neutral=True)
        _apply_group_result(simulated, fixture.home_team, fixture.away_team, hg, ag)

    ranked = _rank_group(simulated)
    return ranked[:2]


def _simulate_ko_match(home: str, away: str, elo_map: dict[str, float]) -> str:
    hg, ag = _simulate_match(home, away, elo_map, neutral=True)
    if hg > ag:
        return home
    elif ag > hg:
        return away
    return _penalty_winner(home, away)


class WorldCupSimulator:
    def __init__(
        self,
        teams: list[str],
        simulations: int = 10000,
        elo_map: dict[str, float] | None = None,
        fixtures: list[TournamentFixture] | None = None,
        standings: list[GroupStandingRecord] | None = None,
        use_current_state: bool = False,
    ) -> None:
        self.teams = teams
        self.simulations = simulations
        self.elo_map = elo_map or {}
        self.fixtures = fixtures or []
        self.standings = standings or []
        self.use_current_state = use_current_state

    def simulate(self) -> TournamentBracket:
        group_advance: dict[str, int] = {t: 0 for t in self.teams}
        qf: dict[str, int] = {t: 0 for t in self.teams}
        sf: dict[str, int] = {t: 0 for t in self.teams}
        final: dict[str, int] = {t: 0 for t in self.teams}
        champion: dict[str, int] = {t: 0 for t in self.teams}

        current_tables, team_to_group = _standings_to_groups(self.standings)
        fixture_groups = _fixtures_by_group(self.fixtures, team_to_group)
        grouped_state = self._build_group_state(current_tables, fixture_groups)

        for _ in range(self.simulations):
            qualifiers: list[str] = []
            for group_name, group_data in grouped_state.items():
                table = group_data["table"]
                fixtures = group_data["fixtures"]
                teams = list(table.keys())
                if len(teams) < 2:
                    qualifiers.extend(teams)
                    continue
                if fixtures or (self.use_current_state and self.standings):
                    adv = _simulate_group_from_state(
                        table,
                        fixtures,
                        self.elo_map,
                        standings_are_current=bool(self.standings and self.use_current_state),
                    )
                else:
                    adv = _simulate_group(teams, self.elo_map)
                qualifiers.extend(adv)
                for t in adv:
                    if t in group_advance:
                        group_advance[t] += 1

            # Remove byes
            qualifiers = [t for t in qualifiers if not t.startswith("__bye")]
            if not qualifiers:
                continue

            # KO rounds
            round_teams = qualifiers[:]
            round_name = "qf"
            while len(round_teams) > 1:
                next_round: list[str] = []
                for i in range(0, len(round_teams) - 1, 2):
                    winner = _simulate_ko_match(round_teams[i], round_teams[i + 1], self.elo_map)
                    next_round.append(winner)
                    if round_name == "qf" and winner in qf:
                        qf[winner] += 1
                    elif round_name == "sf" and winner in sf:
                        sf[winner] += 1
                    elif round_name == "final" and winner in final:
                        final[winner] += 1
                if len(round_teams) % 2 == 1:
                    next_round.append(round_teams[-1])
                round_teams = next_round
                if round_name == "qf":
                    round_name = "sf"
                elif round_name == "sf":
                    round_name = "final"
                else:
                    round_name = "done"

            if round_teams:
                champ = round_teams[0]
                if champ in champion:
                    champion[champ] += 1

        S = self.simulations
        results = [
            TournamentTeamProbability(
                team=t,
                group_advance=group_advance[t] / S,
                quarter_final=qf[t] / S,
                semi_final=sf[t] / S,
                final=final[t] / S,
                champion=champion[t] / S,
            )
            for t in self.teams
        ]
        results.sort(key=lambda r: r.champion, reverse=True)

        return TournamentBracket(
            teams=self.teams,
            simulations=self.simulations,
            results=results,
            group_results={
                group: {
                    "teams": list(data["table"].keys()),
                    "fixtures": len(data["fixtures"]),
                    "current_state_used": bool(self.standings and self.use_current_state),
                }
                for group, data in grouped_state.items()
            },
            fixtures_used=len(self.fixtures),
            current_state_used=bool(self.standings and self.use_current_state),
        )

    def _build_group_state(
        self,
        current_tables: dict[str, dict[str, dict[str, int]]],
        fixture_groups: dict[str, list[TournamentFixture]],
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        if self.use_current_state and current_tables:
            for group, table in current_tables.items():
                grouped[group] = {
                    "table": {team: dict(values) for team, values in table.items()},
                    "fixtures": fixture_groups.get(group, []),
                }
            return grouped

        if fixture_groups:
            for group, fixtures in fixture_groups.items():
                teams = sorted({team for f in fixtures for team in (f.home_team, f.away_team)})
                grouped[group] = {"table": _empty_table(teams), "fixtures": fixtures}
            return grouped

        group_size = 4
        padded = list(self.teams)
        while len(padded) % group_size != 0:
            padded.append(f"__bye_{len(padded)}")
        groups = [padded[i : i + group_size] for i in range(0, len(padded), group_size)]
        for index, teams in enumerate(groups):
            real = [team for team in teams if not team.startswith("__bye")]
            grouped[f"Group {chr(65 + index)}"] = {"table": _empty_table(real), "fixtures": []}
        return grouped
