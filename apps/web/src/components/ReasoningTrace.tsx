import React, { useState } from "react";
import { Bot, ChevronDown, ChevronRight, Cpu, Eye, Lightbulb, Target } from "lucide-react";
import type { AgentReasoning, ReasoningStep } from "../api";

const SIGNAL_LABEL: Record<string, string> = {
  positive: "利好主队",
  neutral: "中性",
  negative: "利好客队",
};

const STEP_META: Record<ReasoningStep["kind"], { label: string; icon: typeof Eye }> = {
  observation: { label: "观察", icon: Eye },
  analysis: { label: "分析", icon: Lightbulb },
  conclusion: { label: "结论", icon: Target },
};

function ReasoningCard({ reasoning, defaultOpen }: { reasoning: AgentReasoning; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <article className={`reasoning-card ${reasoning.signal}`}>
      <button className="reasoning-head" onClick={() => setOpen((v) => !v)}>
        {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        <Bot size={16} />
        <strong>{reasoning.agent}</strong>
        <span className="reasoning-signal">{SIGNAL_LABEL[reasoning.signal] ?? reasoning.signal}</span>
        <span className="reasoning-conf">置信度 {(reasoning.confidence * 100).toFixed(0)}%</span>
        <span className={`reasoning-badge ${reasoning.powered_by}`} title={reasoning.powered_by === "llm" ? "由大模型推理" : "确定性推理（未启用大模型）"}>
          <Cpu size={12} /> {reasoning.powered_by === "llm" ? "LLM" : "规则"}
        </span>
      </button>

      {open && (
        <div className="reasoning-body">
          <ol className="reasoning-steps">
            {reasoning.steps.map((step, index) => {
              const meta = STEP_META[step.kind];
              const Icon = meta.icon;
              return (
                <li key={index} className={`reasoning-step ${step.kind}`}>
                  <span className="step-tag">
                    <Icon size={13} /> {meta.label}
                  </span>
                  <p>{step.content}</p>
                </li>
              );
            })}
          </ol>
          {reasoning.sources.length > 0 && (
            <div className="reasoning-sources">
              来源：
              {reasoning.sources.map((src, index) =>
                src.startsWith("http") ? (
                  <a key={index} href={src} target="_blank" rel="noreferrer">
                    [{index + 1}]
                  </a>
                ) : (
                  <code key={index}>{src}</code>
                ),
              )}
            </div>
          )}
        </div>
      )}
    </article>
  );
}

export function ReasoningTrace({
  reasonings,
  streaming,
}: {
  reasonings: AgentReasoning[];
  streaming: boolean;
}) {
  if (reasonings.length === 0 && !streaming) {
    return <div className="empty">运行预测后，这里会逐个展示每位智能体的推理过程。</div>;
  }
  return (
    <div className="reasoning-trace">
      {reasonings.map((reasoning, index) => (
        <ReasoningCard
          key={`${reasoning.agent}-${index}`}
          reasoning={reasoning}
          defaultOpen={index >= reasonings.length - 1}
        />
      ))}
      {streaming && (
        <div className="reasoning-thinking">
          <Cpu size={16} className="spin" />
          智能体正在推理中…
        </div>
      )}
    </div>
  );
}
