"""Tests for the LangGraph-based forecast graph.

All tests run without an LLM key so every node uses the deterministic
fallback path.  The graph must still produce valid AgentFindings and
correct SSE-compatible state updates.
"""
from __future__ import annotations

import pytest

from worldcup_forecast.graph.graph import build_forecast_graph
from worldcup_forecast.graph.tools import make_tools
from worldcup_forecast.schemas import MatchPredictionRequest


# ── helpers ───────────────────────────────────────────────────────────────────

def _no_llm_settings() -> dict:
    return {
        "base_url": "http://localhost:11434/v1",
        "api_key": "",
        "model": "test-model",
        "temperature": 0.2,
        "timeout_seconds": 10,
        "enabled": False,
    }


def _no_search_settings() -> dict:
    return {
        "provider": "none",
        "base_url": "",
        "api_key": "",
        "timeout_seconds": 10,
        "max_results": 5,
        "enabled": False,
    }


def _initial_state(home="Brazil", away="Germany") -> dict:
    req = MatchPredictionRequest(home_team=home, away_team=away)
    return {
        "request": req.model_dump(),
        "llm_settings": _no_llm_settings(),
        "search_settings": _no_search_settings(),
        "agent_findings": [],
    }


# ── graph construction ────────────────────────────────────────────────────────

def test_build_forecast_graph_no_store():
    graph = build_forecast_graph(store=None)
    assert graph is not None


def test_build_forecast_graph_with_tmp_store(tmp_store):
    graph = build_forecast_graph(store=tmp_store)
    assert graph is not None


# ── tool make_tools ───────────────────────────────────────────────────────────

def test_make_tools_returns_four_tools(tmp_store):
    tools = make_tools(tmp_store)
    assert set(tools.keys()) == {"query_elo", "get_recent_form", "search_news", "get_odds"}


def test_query_elo_tool(tmp_store):
    tools = make_tools(tmp_store)
    result = tools["query_elo"].invoke({"home_team": "Brazil", "away_team": "Germany"})
    assert "home_elo" in result
    assert "away_elo" in result
    assert "elo_diff" in result
    assert isinstance(result["home_elo"], (int, float))


def test_get_recent_form_empty_store(tmp_store):
    tools = make_tools(tmp_store)
    result = tools["get_recent_form"].invoke({"team": "Brazil", "limit": 5})
    # No intl data → returns note about empty table
    assert "matches" in result or "note" in result


def test_get_recent_form_with_data(tmp_store):
    from datetime import date
    rows = [
        {"date": date(2024, 1, i + 1), "home_team": "Brazil", "away_team": "Chile",
         "home_score": 2, "away_score": 0, "tournament": "Friendly", "neutral": False}
        for i in range(5)
    ]
    tmp_store.insert_intl_results(rows)
    tools = make_tools(tmp_store)
    result = tools["get_recent_form"].invoke({"team": "Brazil", "limit": 5})
    assert result["matches"] == 5
    assert result["wins"] == 5
    assert result["points_per_game"] == 3.0


def test_search_news_disabled():
    tools = make_tools(None, search_settings=None)
    result = tools["search_news"].invoke({"query": "Brazil injury news"})
    assert result["ok"] is False


def test_get_odds_no_match(tmp_store):
    tools = make_tools(tmp_store)
    result = tools["get_odds"].invoke({"home_team": "Brazil", "away_team": "Germany"})
    assert result["found"] is False


# ── supervisor node ───────────────────────────────────────────────────────────

def test_supervisor_node_populates_stats(tmp_store):
    from worldcup_forecast.graph.nodes import supervisor_node
    state = _initial_state()
    state["_store"] = tmp_store
    state["_tools"] = {}
    output = supervisor_node(state)
    assert "probabilities" in output
    probs = output["probabilities"]
    assert abs(probs["home_win"] + probs["draw"] + probs["away_win"] - 1.0) < 0.01
    assert "bet_signals" in output
    assert "expected_score" in output
    assert "most_likely_score" in output


# ── full graph ainvoke (deterministic path) ───────────────────────────────────

@pytest.mark.asyncio
async def test_graph_ainvoke_deterministic(tmp_store):
    graph = build_forecast_graph(store=tmp_store)
    state = _initial_state()
    final = await graph.ainvoke(state)

    # probabilities should sum to ~1
    probs = final["probabilities"]
    assert abs(probs["home_win"] + probs["draw"] + probs["away_win"] - 1.0) < 0.01

    # should have findings from all 7 agents
    findings = final.get("agent_findings", [])
    assert len(findings) >= 4, f"Expected ≥4 agent findings, got {len(findings)}"

    # every finding must have required fields
    for f in findings:
        assert f["signal"] in ("positive", "neutral", "negative")
        assert 0.0 <= f["confidence"] <= 1.0
        assert isinstance(f["rationale"], str) and len(f["rationale"]) > 0

    # report fields
    assert isinstance(final.get("explanation"), str)
    assert isinstance(final.get("report_id"), str)


@pytest.mark.asyncio
async def test_graph_ainvoke_all_agent_names(tmp_store):
    graph = build_forecast_graph(store=tmp_store)
    final = await graph.ainvoke(_initial_state())
    agent_names = {f["agent"] for f in final.get("agent_findings", [])}
    assert "实力分析员" in agent_names
    assert "近期状态分析员" in agent_names
    assert "新闻舆情分析员" in agent_names
    assert "赔率市场分析员" in agent_names
    assert "正方研究员" in agent_names
    assert "反方研究员" in agent_names
    assert "风控经理" in agent_names


@pytest.mark.asyncio
async def test_graph_debate_bear_differs_from_bull(tmp_store):
    """Bear's rationale should not be identical to bull's (real two-turn debate or distinct deterministic text)."""
    graph = build_forecast_graph(store=tmp_store)
    final = await graph.ainvoke(_initial_state())
    findings = {f["agent"]: f for f in final.get("agent_findings", [])}
    bull = findings.get("正方研究员", {})
    bear = findings.get("反方研究员", {})
    assert bull.get("rationale") != bear.get("rationale")
    assert bull.get("signal") == "positive"
    assert bear.get("signal") == "negative"


# ── streaming events ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_graph_astream_events_emits_supervisor(tmp_store):
    graph = build_forecast_graph(store=tmp_store)
    found_supervisor = False
    async for event in graph.astream_events(_initial_state(), version="v2"):
        if event.get("event") == "on_chain_end" and event.get("name") == "supervisor":
            found_supervisor = True
            output = event["data"]["output"]
            assert "probabilities" in output
            break
    assert found_supervisor, "supervisor on_chain_end event not emitted"


@pytest.mark.asyncio
async def test_graph_astream_events_emits_analysis_nodes(tmp_store):
    graph = build_forecast_graph(store=tmp_store)
    analysis_nodes_seen = set()
    async for event in graph.astream_events(_initial_state(), version="v2"):
        if event.get("event") == "on_chain_end":
            name = event.get("name", "")
            if name in {"strength_node", "form_node", "news_node", "odds_node"}:
                analysis_nodes_seen.add(name)
    assert analysis_nodes_seen == {"strength_node", "form_node", "news_node", "odds_node"}


@pytest.mark.asyncio
async def test_graph_astream_events_report_node_last(tmp_store):
    graph = build_forecast_graph(store=tmp_store)
    events_order = []
    async for event in graph.astream_events(_initial_state(), version="v2"):
        if event.get("event") == "on_chain_end":
            name = event.get("name", "")
            if name in {"supervisor", "debate_node", "risk_node", "report_node"}:
                events_order.append(name)
    assert "supervisor" in events_order
    assert "report_node" in events_order
    # report_node must come after supervisor
    assert events_order.index("report_node") > events_order.index("supervisor")
