from __future__ import annotations

import pytest
from worldcup_forecast.tournament import WorldCupSimulator
from worldcup_forecast.schemas import GroupStandingRecord, TournamentFixture


TEAMS_4 = ["Brazil", "Argentina", "France", "Germany"]
TEAMS_8 = ["Brazil", "Argentina", "France", "Germany", "England", "Spain", "Portugal", "Netherlands"]


def test_probabilities_in_range():
    sim = WorldCupSimulator(TEAMS_4, simulations=200)
    bracket = sim.simulate()
    for r in bracket.results:
        assert 0.0 <= r.group_advance <= 1.0
        assert 0.0 <= r.quarter_final <= 1.0
        assert 0.0 <= r.semi_final <= 1.0
        assert 0.0 <= r.final <= 1.0
        assert 0.0 <= r.champion <= 1.0


def test_champion_probs_sum_to_one():
    sim = WorldCupSimulator(TEAMS_4, simulations=500)
    bracket = sim.simulate()
    total = sum(r.champion for r in bracket.results)
    assert abs(total - 1.0) < 0.05


def test_all_teams_returned():
    sim = WorldCupSimulator(TEAMS_8, simulations=200)
    bracket = sim.simulate()
    returned = {r.team for r in bracket.results}
    assert returned == set(TEAMS_8)


def test_reproducible_with_seed():
    import random
    random.seed(99)
    b1 = WorldCupSimulator(TEAMS_4, simulations=300).simulate()
    random.seed(99)
    b2 = WorldCupSimulator(TEAMS_4, simulations=300).simulate()
    for r1, r2 in zip(b1.results, b2.results):
        assert r1.team == r2.team
        assert abs(r1.champion - r2.champion) < 1e-9


def test_sorted_by_champion_desc():
    sim = WorldCupSimulator(TEAMS_8, simulations=300)
    bracket = sim.simulate()
    champs = [r.champion for r in bracket.results]
    assert champs == sorted(champs, reverse=True)


def test_two_teams():
    sim = WorldCupSimulator(["Brazil", "Germany"], simulations=500)
    bracket = sim.simulate()
    total = sum(r.champion for r in bracket.results)
    assert abs(total - 1.0) < 0.05


def test_current_standings_are_used_for_group_qualification():
    standings = [
        GroupStandingRecord(group_name="Group A", rank=1, team="Brazil", points=7, goal_difference=5, goals_for=6),
        GroupStandingRecord(group_name="Group A", rank=2, team="Germany", points=6, goal_difference=2, goals_for=4),
        GroupStandingRecord(group_name="Group A", rank=3, team="France", points=1, goal_difference=-2, goals_for=2),
        GroupStandingRecord(group_name="Group A", rank=4, team="Argentina", points=0, goal_difference=-5, goals_for=1),
    ]

    bracket = WorldCupSimulator(
        ["Brazil", "Germany", "France", "Argentina"],
        simulations=100,
        standings=standings,
        use_current_state=True,
    ).simulate()
    by_team = {row.team: row for row in bracket.results}

    assert by_team["Brazil"].group_advance == 1
    assert by_team["Germany"].group_advance == 1
    assert by_team["France"].group_advance == 0
    assert by_team["Argentina"].group_advance == 0
    assert bracket.current_state_used is True


def test_fixture_schedule_defines_groups():
    fixtures = [
        TournamentFixture(match_id="A1", group_name="Group A", home_team="Brazil", away_team="Germany"),
        TournamentFixture(match_id="A2", group_name="Group A", home_team="France", away_team="Argentina"),
        TournamentFixture(match_id="B1", group_name="Group B", home_team="England", away_team="Spain"),
        TournamentFixture(match_id="B2", group_name="Group B", home_team="Portugal", away_team="Netherlands"),
    ]

    bracket = WorldCupSimulator(
        ["Brazil", "Germany", "France", "Argentina", "England", "Spain", "Portugal", "Netherlands"],
        simulations=100,
        fixtures=fixtures,
    ).simulate()

    assert bracket.fixtures_used == 4
    assert set(bracket.group_results) == {"Group A", "Group B"}
