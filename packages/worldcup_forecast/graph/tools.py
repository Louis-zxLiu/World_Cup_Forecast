"""LangGraph tool definitions for the World Cup Forecast agents.

Each tool wraps a deterministic data-access function so the LLM can call it
via the tool-calling API.  Use ``make_tools()`` to get a store-bound list.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from ..storage import ForecastStore
    from ..schemas import SearchSettings


def make_tools(store: "ForecastStore", search_settings: "SearchSettings | None" = None):
    """Return store-bound tool instances for injection into agent nodes."""

    @tool
    def query_elo(home_team: str, away_team: str) -> dict:
        """查询两支球队的 Elo 评分和实力差。返回 home_elo、away_elo、elo_diff。"""
        from ..modeling import BASE_ELO, NEUTRAL_ELO
        elo_map: dict = store.get_team_elo() if store else {}
        home_elo = elo_map.get(home_team, BASE_ELO.get(home_team, NEUTRAL_ELO))
        away_elo = elo_map.get(away_team, BASE_ELO.get(away_team, NEUTRAL_ELO))
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": round(home_elo, 1),
            "away_elo": round(away_elo, 1),
            "elo_diff": round(home_elo - away_elo, 1),
            "source": "internal:team_elo" if elo_map else "internal:baseline_elo",
        }

    @tool
    def get_recent_form(team: str, limit: int = 10) -> dict:
        """获取球队近 N 场国际比赛的胜平负、场均积分（PPG）和净胜球。"""
        from ..form import team_form
        if store is None or store.intl_count() == 0:
            return {"team": team, "matches": 0, "note": "intl_results 表为空，请先初始化数据"}
        fs = team_form(store, team, limit=limit)
        return {
            "team": team,
            "matches": fs.matches,
            "wins": fs.wins,
            "draws": fs.draws,
            "losses": fs.losses,
            "goals_for": fs.goals_for,
            "goals_against": fs.goals_against,
            "points_per_game": round(fs.points_per_game, 3),
            "goal_diff_per_game": round(fs.goal_diff_per_game, 3),
            "form_string": fs.form_string(),
        }

    @tool
    def search_news(query: str) -> dict:
        """搜索球队最新伤停/阵容新闻。返回标题列表和是否发现伤停信号。"""
        if search_settings is None or not search_settings.enabled or not search_settings.api_key:
            return {"ok": False, "hits": [], "note": "联网搜索未配置或未启用"}
        from ..search import WebSearchProvider
        provider = WebSearchProvider(search_settings)
        outcome = provider.search_sync(query)
        return {
            "ok": outcome.ok,
            "hits": [{"title": h.title, "snippet": h.snippet, "url": h.url} for h in outcome.hits],
            "error": outcome.error or "",
            "count": len(outcome.hits),
        }

    @tool
    def get_odds(home_team: str, away_team: str) -> dict:
        """获取该场比赛的 500彩票网胜平负赔率数据。"""
        if store is None:
            return {"found": False, "note": "store 未初始化"}
        odds = store.find_match_odds(home_team, away_team)
        if odds is None:
            return {"found": False, "home_team": home_team, "away_team": away_team}
        return {
            "found": True,
            "home_team": home_team,
            "away_team": away_team,
            "play_type": odds.play_type,
            "win_odds": odds.win_odds,
            "draw_odds": odds.draw_odds,
            "lose_odds": odds.lose_odds,
            "source": odds.source,
            "source_url": odds.source_url,
        }

    return {
        "query_elo": query_elo,
        "get_recent_form": get_recent_form,
        "search_news": search_news,
        "get_odds": get_odds,
    }
