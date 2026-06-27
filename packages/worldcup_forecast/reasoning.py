"""Expose each agent's chain of reasoning.

Every analysis agent already produces a deterministic :class:`AgentFinding`
(its signal, confidence and the metrics that justify them). This module turns
that finding into an inspectable, step-by-step reasoning trace. When an LLM is
configured, it narrates the evidence into observation/analysis/conclusion
steps; otherwise a deterministic fallback builds the same structure directly
from the metrics so the feature works offline.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from .llm import OpenAICompatibleClient
from .schemas import (
    AgentFinding,
    AgentReasoning,
    LLMSettings,
    MatchPredictionRequest,
    ReasoningStep,
)

_SIGNAL_LABEL = {"positive": "利好主队", "neutral": "中性", "negative": "利好客队"}

_SYSTEM_PROMPT = (
    "你是世界杯预测系统里的一名专业分析智能体。你会拿到本智能体的角色、"
    "已经算好的结论（信号与置信度）以及支撑用的结构化数据指标。"
    "请把你的推理过程拆成 2-4 个步骤，每步是一个 JSON 对象，"
    "kind 取值为 observation（观察到的数据事实）、analysis（基于数据的推断）、"
    "conclusion（落到信号上的结论）。只输出一个 JSON 数组，不要多余文字。"
    "每步 content 用简洁中文，引用具体数字，不要编造数据里没有的信息。"
)


def _finding_to_prompt(agent_role: str, finding: AgentFinding, request: MatchPredictionRequest) -> str:
    return (
        f"比赛：{request.home_team}（主） vs {request.away_team}（客）\n"
        f"智能体角色：{agent_role}\n"
        f"已得结论信号：{finding.signal}（{_SIGNAL_LABEL.get(finding.signal, finding.signal)}）\n"
        f"置信度：{finding.confidence}\n"
        f"支撑指标：{json.dumps(finding.metrics, ensure_ascii=False)}\n"
        f"一句话依据：{finding.rationale}"
    )


def deterministic_steps(finding: AgentFinding) -> list[ReasoningStep]:
    """Build a reasoning trace from a finding's own data, without an LLM."""
    steps: list[ReasoningStep] = []
    if finding.metrics:
        metric_text = "，".join(
            f"{key}={value}" for key, value in finding.metrics.items()
        )
        steps.append(ReasoningStep(kind="observation", content=f"采集到的关键指标：{metric_text}。"))
    else:
        steps.append(ReasoningStep(kind="observation", content="本维度暂无结构化指标可观察。"))
    steps.append(ReasoningStep(kind="analysis", content=finding.rationale))
    steps.append(
        ReasoningStep(
            kind="conclusion",
            content=(
                f"综合判断信号为「{_SIGNAL_LABEL.get(finding.signal, finding.signal)}」，"
                f"置信度 {finding.confidence:.0%}。"
            ),
        )
    )
    return steps


def _parse_llm_steps(text: str) -> list[ReasoningStep]:
    """Parse the LLM's JSON array of steps, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned.strip("`")
        cleaned = cleaned.lstrip("json").strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array found")
    payload = json.loads(cleaned[start : end + 1])
    steps: list[ReasoningStep] = []
    for item in payload:
        kind = item.get("kind", "analysis")
        if kind not in ("observation", "analysis", "conclusion"):
            kind = "analysis"
        content = str(item.get("content", "")).strip()
        if content:
            steps.append(ReasoningStep(kind=kind, content=content))
    if not steps:
        raise ValueError("empty steps")
    return steps


async def reason_for_finding(
    agent_role: str,
    finding: AgentFinding,
    request: MatchPredictionRequest,
    settings: LLMSettings,
) -> AgentReasoning:
    """Produce a structured reasoning trace for a single agent finding."""
    powered_by = "deterministic"
    steps: list[ReasoningStep]
    if settings.enabled and settings.api_key:
        try:
            client = OpenAICompatibleClient(settings)
            raw = await client.complete(
                _SYSTEM_PROMPT, _finding_to_prompt(agent_role, finding, request)
            )
            steps = _parse_llm_steps(raw)
            powered_by = "llm"
        except Exception:
            steps = deterministic_steps(finding)
    else:
        steps = deterministic_steps(finding)
    return AgentReasoning(
        agent=finding.agent,
        confidence=finding.confidence,
        signal=finding.signal,
        steps=steps,
        rationale=finding.rationale,
        sources=finding.sources,
        metrics=finding.metrics,
        powered_by=powered_by,
    )
