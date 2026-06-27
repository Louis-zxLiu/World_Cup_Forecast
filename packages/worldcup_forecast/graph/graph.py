"""Build and compile the LangGraph forecast graph.

Usage::

    graph = build_forecast_graph(store, search_settings)
    result_state = await graph.ainvoke(initial_state)

    # or streaming:
    async for event in graph.astream_events(initial_state, version="v2"):
        ...
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .nodes import (
    debate_node,
    form_node,
    news_node,
    odds_node,
    report_node,
    risk_node,
    strength_node,
    supervisor_node,
)
from .state import ForecastState

if TYPE_CHECKING:
    from ..schemas import SearchSettings
    from ..storage import ForecastStore

# Names of the 4 parallel analysis nodes (used for fan-in edge wiring).
_ANALYSIS_NODES = ["strength_node", "form_node", "news_node", "odds_node"]


def build_forecast_graph(
    store: "ForecastStore | None" = None,
    search_settings: "SearchSettings | None" = None,
):
    """Return a compiled LangGraph StateGraph.

    ``store`` and ``search_settings`` are injected into every node via the
    ``_store`` and ``_tools`` keys in the state dict.  Callers must include
    these in ``initial_state`` when invoking the graph.
    """
    from .tools import make_tools

    tool_map = make_tools(store, search_settings) if store is not None else {}

    # ── inject runtime dependencies into each node via closure ────────────────
    def _inject(node_fn):
        """Wrap a node so it always receives _store and _tools in state."""
        import functools

        @functools.wraps(node_fn)
        async def wrapper(state: dict):
            state = dict(state)
            state["_store"] = store
            state["_tools"] = tool_map
            if hasattr(node_fn, "__wrapped__") or not _is_async(node_fn):
                return node_fn(state)
            return await node_fn(state)

        return wrapper

    def _inject_sync(node_fn):
        import functools

        @functools.wraps(node_fn)
        def wrapper(state: dict):
            state = dict(state)
            state["_store"] = store
            state["_tools"] = tool_map
            return node_fn(state)

        return wrapper

    def _is_async(fn):
        import asyncio
        return asyncio.iscoroutinefunction(fn)

    # ── build graph ───────────────────────────────────────────────────────────
    builder = StateGraph(ForecastState)

    builder.add_node("supervisor", _inject_sync(supervisor_node))
    builder.add_node("strength_node", _inject(strength_node))
    builder.add_node("form_node", _inject(form_node))
    builder.add_node("news_node", _inject(news_node))
    builder.add_node("odds_node", _inject(odds_node))
    builder.add_node("debate_node", _inject(debate_node))
    builder.add_node("risk_node", _inject_sync(risk_node))
    builder.add_node("report_node", _inject(report_node))

    # supervisor → fan-out to 4 parallel analysis nodes
    builder.add_edge(START, "supervisor")
    for node in _ANALYSIS_NODES:
        builder.add_edge("supervisor", node)

    # fan-in: all 4 analysis nodes must finish before debate
    for node in _ANALYSIS_NODES:
        builder.add_edge(node, "debate_node")

    builder.add_edge("debate_node", "risk_node")
    builder.add_edge("risk_node", "report_node")
    builder.add_edge("report_node", END)

    return builder.compile()
