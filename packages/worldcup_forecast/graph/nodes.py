"""LangGraph nodes for the World Cup Forecast multi-agent graph.

Each node receives the full ForecastState and returns a partial state update.
ReAct agents (strength/form/news/odds) use LangGraph's create_react_agent
when LLM is configured; they fall back to deterministic logic otherwise.
"""
from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..storage import ForecastStore
    from ..schemas import LLMSettings, SearchSettings

_FINDING_SCHEMA = """{
  "agent": "<必须使用下面指定的中文角色名，不得修改>",
  "confidence": 0.0-1.0,
  "signal": "positive|neutral|negative",
  "rationale": "<两三句中文分析依据，引用具体数字，不要暴露原始字段名或 Python dict>",
  "sources": ["<来源>"],
  "metrics": {}
}"""

_AGENT_SYSTEM = (
    "你是世界杯预测系统中的专项分析智能体。"
    "使用提供的工具收集数据，然后输出一个 JSON 对象（无多余文字）：\n"
    + _FINDING_SCHEMA
    + "\nsignal 的判断依据：positive=利好主队，neutral=无明显倾向，negative=利好客队。"
    "\n重要：agent 字段必须使用任务开头指定的中文角色名，不得使用英文或自行命名。"
    "\nrationale 必须用自然中文写成，不得输出 Python dict、字段名或原始数据结构。"
)


def _make_llm(llm_settings: dict):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=llm_settings["model"],
        base_url=llm_settings["base_url"],
        api_key=llm_settings["api_key"],
        temperature=llm_settings.get("temperature", 0.2),
        timeout=llm_settings.get("timeout_seconds", 30),
    )


def _llm_enabled(llm_settings: dict) -> bool:
    return bool(llm_settings.get("enabled") and llm_settings.get("api_key"))


def _parse_finding(text: str, agent_name: str) -> dict:
    """Extract JSON finding from LLM output, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 2 else parts[-1]
        cleaned = cleaned.lstrip("json").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1:
        try:
            finding = json.loads(cleaned[start:end + 1])
            finding.setdefault("agent", agent_name)
            finding.setdefault("sources", [])
            finding.setdefault("metrics", {})
            if finding.get("signal") not in ("positive", "neutral", "negative"):
                finding["signal"] = "neutral"
            finding["confidence"] = max(0.0, min(1.0, float(finding.get("confidence", 0.5))))
            return finding
        except (json.JSONDecodeError, ValueError):
            pass
    return {
        "agent": agent_name,
        "confidence": 0.4,
        "signal": "neutral",
        "rationale": text[:300],
        "sources": ["llm:raw"],
        "metrics": {},
    }


async def _run_react_agent(
    agent_name: str,
    task_prompt: str,
    tools: list,
    llm_settings: dict,
) -> dict:
    """Run a ReAct agent with the given tools; return an AgentFinding dict."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.prebuilt import create_react_agent

    llm = _make_llm(llm_settings)
    agent = create_react_agent(llm, tools)
    messages = [
        SystemMessage(content=_AGENT_SYSTEM),
        HumanMessage(content=task_prompt),
    ]
    result = await agent.ainvoke({"messages": messages})
    last_msg = result["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    return _parse_finding(content, agent_name)


# ── supervisor ─────────────────────────────────────────────────────────────────

def supervisor_node(state: dict) -> dict:
    """Run deterministic stats (Elo/Poisson) and populate base prediction fields."""
    from ..modeling import BaselineForecastModel, build_bet_signals
    from ..schemas import MatchPredictionRequest

    req = MatchPredictionRequest(**state["request"])
    # store is injected via closure in build_forecast_graph
    _store = state.get("_store")
    model = BaselineForecastModel(elo_map=_store.get_team_elo() if _store else None, store=_store)
    dist = model.predict_score_distribution(req)

    odds = _store.find_match_odds(req.home_team, req.away_team) if _store else None
    bet_signals = build_bet_signals(req, dist.probabilities, odds)

    shap_values: dict = {}
    try:
        from ..ml_model import SHAPExplainer, XGBoostForecastModel
        xgb = XGBoostForecastModel()
        if xgb._clf is not None:
            shap_values = SHAPExplainer(xgb).top_features(req.home_team, req.away_team, req.neutral_site)
    except Exception:
        pass

    return {
        "probabilities": dist.probabilities.model_dump(),
        "expected_score": (round(dist.expected_home_goals, 3), round(dist.expected_away_goals, 3)),
        "most_likely_score": dist.most_likely_score,
        "bet_signals": [s.model_dump() for s in bet_signals],
        "odds": odds.model_dump() if odds else None,
        "shap_values": shap_values,
        "agent_findings": [],
    }


# ── ReAct analysis nodes ───────────────────────────────────────────────────────

async def strength_node(state: dict) -> dict:
    from ..agents import StrengthAgent
    from ..schemas import MatchPredictionRequest

    req = MatchPredictionRequest(**state["request"])
    llm_settings = state.get("llm_settings", {})
    _store = state.get("_store")

    if not _llm_enabled(llm_settings):
        finding = StrengthAgent(_store).analyze(req).model_dump()
        return {"agent_findings": [finding]}

    tool_map = state["_tools"]
    task = (
        f"角色名（必须原样填入 agent 字段）：实力分析员\n"
        f"分析 {req.home_team}（主场）vs {req.away_team}（客场）的实力对比。\n"
        f"请调用 query_elo 工具获取双方 Elo 评分，然后给出分析结论。"
    )
    finding = await _run_react_agent("实力分析员", task, [tool_map["query_elo"]], llm_settings)
    return {"agent_findings": [finding]}


async def form_node(state: dict) -> dict:
    from ..agents import FormAgent
    from ..schemas import MatchPredictionRequest

    req = MatchPredictionRequest(**state["request"])
    llm_settings = state.get("llm_settings", {})
    _store = state.get("_store")

    if not _llm_enabled(llm_settings):
        finding = FormAgent(_store).analyze(req).model_dump()
        return {"agent_findings": [finding]}

    tool_map = state["_tools"]
    task = (
        f"角色名（必须原样填入 agent 字段）：近期状态分析员\n"
        f"分析 {req.home_team} 和 {req.away_team} 的近期状态。\n"
        f"请分别调用 get_recent_form 工具获取两队近10场数据，对比场均积分和净胜球，给出中文结论。"
    )
    finding = await _run_react_agent("近期状态分析员", task, [tool_map["get_recent_form"]], llm_settings)
    return {"agent_findings": [finding]}


async def news_node(state: dict) -> dict:
    from ..agents import NewsSentimentAgent
    from ..schemas import MatchPredictionRequest, SearchSettings

    req = MatchPredictionRequest(**state["request"])
    llm_settings = state.get("llm_settings", {})
    search_cfg = state.get("search_settings", {})
    _store = state.get("_store")

    search_settings = SearchSettings(**search_cfg) if search_cfg else None

    if not _llm_enabled(llm_settings):
        finding = NewsSentimentAgent(search_settings).analyze(req).model_dump()
        return {"agent_findings": [finding]}

    tool_map = state["_tools"]
    task = (
        f"角色名（必须原样填入 agent 字段）：新闻舆情分析员\n"
        f"搜索 {req.home_team} 和 {req.away_team} 的最新伤停、阵容消息。\n"
        f"分别搜索两队，判断是否有影响比赛结果的伤停/缺阵信号，给出中文结论。不要在 rationale 里输出原始字段名或数据结构。"
    )
    finding = await _run_react_agent("新闻舆情分析员", task, [tool_map["search_news"]], llm_settings)
    return {"agent_findings": [finding]}


async def odds_node(state: dict) -> dict:
    from ..agents import OddsMarketAgent
    from ..schemas import MatchPredictionRequest

    req = MatchPredictionRequest(**state["request"])
    llm_settings = state.get("llm_settings", {})
    _store = state.get("_store")

    if not _llm_enabled(llm_settings):
        odds = _store.find_match_odds(req.home_team, req.away_team) if _store else None
        finding = OddsMarketAgent().analyze(odds).model_dump()
        return {"agent_findings": [finding]}

    tool_map = state["_tools"]
    task = (
        f"角色名（必须原样填入 agent 字段）：赔率市场分析员\n"
        f"获取 {req.home_team} vs {req.away_team} 的赔率数据。\n"
        f"调用 get_odds 工具，分析胜/平/负赔率隐含的市场概率，给出中文结论。不要在 rationale 里输出原始字段名或数据结构。"
    )
    finding = await _run_react_agent("赔率市场分析员", task, [tool_map["get_odds"]], llm_settings)
    return {"agent_findings": [finding]}


# ── debate node ────────────────────────────────────────────────────────────────

async def debate_node(state: dict) -> dict:
    """Bull/bear debate: bull argues first, bear sees bull's argument then rebuts."""
    from ..agents import BullBearDebateAgents
    from ..schemas import MatchPredictionRequest, PredictionResult, OutcomeProbability, BetSignal

    req = MatchPredictionRequest(**state["request"])
    llm_settings = state.get("llm_settings", {})
    findings = state.get("agent_findings", [])
    probs_dict = state.get("probabilities", {"home_win": 0.33, "draw": 0.33, "away_win": 0.34})
    bet_signals_raw = state.get("bet_signals", [])

    if not _llm_enabled(llm_settings):
        # Deterministic fallback: reconstruct PredictionResult-like object
        from ..schemas import OutcomeProbability, BetSignal, AgentFinding
        prob = OutcomeProbability(**probs_dict)
        signals = [BetSignal(**s) for s in bet_signals_raw]
        agent_findings_objs = []
        for f in findings:
            try:
                agent_findings_objs.append(AgentFinding(**f))
            except Exception:
                pass

        dummy_result = PredictionResult(
            match=req,
            probabilities=prob,
            expected_score=state.get("expected_score", (1.3, 1.1)),
            most_likely_score=state.get("most_likely_score", "1-1"),
            bet_signals=signals,
            agent_findings=agent_findings_objs,
            explanation="",
        )
        debate_findings = BullBearDebateAgents().debate(dummy_result)
        return {
            "agent_findings": [f.model_dump() for f in debate_findings],
            "bull_argument": debate_findings[0].rationale if debate_findings else "",
            "bear_argument": debate_findings[1].rationale if len(debate_findings) > 1 else "",
        }

    llm = _make_llm(llm_settings)
    findings_text = json.dumps(findings, ensure_ascii=False, indent=2)
    probs_text = json.dumps(probs_dict, ensure_ascii=False)
    bet_text = json.dumps(bet_signals_raw, ensure_ascii=False)

    # Round 1: bull argues
    from langchain_core.messages import HumanMessage, SystemMessage
    bull_prompt = (
        f"比赛：{req.home_team}（主）vs {req.away_team}（客）\n"
        f"概率：{probs_text}\n投注信号：{bet_text}\n"
        f"各智能体分析：{findings_text}\n\n"
        "你是正方研究员，请基于以上数据给出支持投注主队的最强论点（2-3句）。"
    )
    bull_resp = await llm.ainvoke([SystemMessage(content="你是足球量化分析师"), HumanMessage(content=bull_prompt)])
    bull_arg = bull_resp.content

    # Round 2: bear sees bull's argument then rebuts
    bear_prompt = (
        f"比赛：{req.home_team}（主）vs {req.away_team}（客）\n"
        f"概率：{probs_text}\n投注信号：{bet_text}\n"
        f"正方论点：{bull_arg}\n\n"
        "你是反方研究员，请针对正方论点给出反驳意见，指出模型风险和不确定性（2-3句）。"
    )
    bear_resp = await llm.ainvoke([SystemMessage(content="你是足球量化风控分析师"), HumanMessage(content=bear_prompt)])
    bear_arg = bear_resp.content

    bull_finding = {
        "agent": "正方研究员",
        "confidence": 0.6,
        "signal": "positive",
        "rationale": bull_arg,
        "sources": ["internal:model_probability", "internal:agent_findings"],
        "metrics": {"home_win_prob": probs_dict.get("home_win", 0), "draw_prob": probs_dict.get("draw", 0)},
    }
    bear_finding = {
        "agent": "反方研究员",
        "confidence": 0.65,
        "signal": "negative",
        "rationale": bear_arg,
        "sources": ["internal:model_risk"],
        "metrics": {"away_win_prob": probs_dict.get("away_win", 0)},
    }
    return {
        "agent_findings": [bull_finding, bear_finding],
        "bull_argument": bull_arg,
        "bear_argument": bear_arg,
    }


# ── risk node ──────────────────────────────────────────────────────────────────

def risk_node(state: dict) -> dict:
    from ..agents import RiskManagerAgent
    from ..schemas import (
        MatchPredictionRequest, PredictionResult, OutcomeProbability,
        BetSignal, AgentFinding,
    )

    req = MatchPredictionRequest(**state["request"])
    probs = OutcomeProbability(**state.get("probabilities", {"home_win": 0.33, "draw": 0.33, "away_win": 0.34}))
    signals = [BetSignal(**s) for s in state.get("bet_signals", [])]
    agent_findings_objs = []
    for f in state.get("agent_findings", []):
        try:
            agent_findings_objs.append(AgentFinding(**f))
        except Exception:
            pass

    dummy_result = PredictionResult(
        match=req,
        probabilities=probs,
        expected_score=state.get("expected_score", (1.3, 1.1)),
        most_likely_score=state.get("most_likely_score", "1-1"),
        bet_signals=signals,
        agent_findings=agent_findings_objs,
        explanation="",
    )
    finding = RiskManagerAgent().analyze(dummy_result)
    return {"agent_findings": [finding.model_dump()]}


# ── report node ────────────────────────────────────────────────────────────────

async def report_node(state: dict) -> dict:
    from ..agents import template_explanation
    from ..schemas import (
        MatchPredictionRequest, PredictionResult, OutcomeProbability,
        BetSignal, AgentFinding,
    )

    req = MatchPredictionRequest(**state["request"])
    llm_settings = state.get("llm_settings", {})
    probs = OutcomeProbability(**state.get("probabilities", {"home_win": 0.33, "draw": 0.33, "away_win": 0.34}))
    signals = [BetSignal(**s) for s in state.get("bet_signals", [])]
    agent_findings_objs = []
    for f in state.get("agent_findings", []):
        try:
            agent_findings_objs.append(AgentFinding(**f))
        except Exception:
            pass

    result = PredictionResult(
        match=req,
        probabilities=probs,
        expected_score=state.get("expected_score", (1.3, 1.1)),
        most_likely_score=state.get("most_likely_score", "1-1"),
        bet_signals=signals,
        agent_findings=agent_findings_objs,
        explanation="",
    )

    explanation = template_explanation(result)
    if _llm_enabled(llm_settings):
        try:
            from ..llm import OpenAICompatibleClient
            from ..schemas import LLMSettings
            settings = LLMSettings(**llm_settings)
            prompt = (
                "请用中文生成一份简洁、可解释、带风险提示的世界杯单场预测报告。"
                f"数据如下：{result.model_dump_json()}"
            )
            explanation = await OpenAICompatibleClient(settings).complete(
                "你是谨慎的足球量化和投注风控分析师，不承诺稳赚。", prompt
            )
        except Exception:
            pass

    report_id = str(uuid.uuid4())
    _store = state.get("_store")
    if _store:
        _store.save_report(report_id, result.model_dump_json())

    return {"explanation": explanation, "report_id": report_id}
