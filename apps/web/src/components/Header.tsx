import React from "react";
import { BrainCircuit, Database, RefreshCw, Server, Sparkles } from "lucide-react";
import type { Health } from "../api";

type Props = {
  health: Health | null;
  onRefresh: () => void;
};

export function Header({ health, onRefresh }: Props) {
  const online = health?.status === "ok";
  return (
    <header className="topbar">
      <div className="topbar-brand">
        <BrainCircuit size={34} />
        <div>
          <h1>世界杯量化预测研究台</h1>
          <span>多智能体解释、赔率 Edge、Kelly 仓位、参数回测</span>
        </div>
      </div>

      <div className="status-strip">
        <span className={`status-badge ${online ? "online" : "warn"}`}>
          <Server size={16} />
          {online ? "后端在线" : "后端未连接"}
        </span>
        {health && (
          <>
            <span className="status-badge">
              <Database size={16} />
              赔率 {health.odds_records}
            </span>
            <span className="status-badge">
              <Database size={16} />
              赛程 {health.live_matches}
            </span>
            <span className="status-badge">
              <Database size={16} />
              积分榜 {health.standing_records}
            </span>
            <span className={`status-badge ${health.llm_enabled ? "online" : ""}`}>
              <Sparkles size={16} />
              {health.llm_enabled ? "LLM 已启用" : "LLM 未启用"}
            </span>
            <span className="status-badge status-model" title={health.model_version}>
              {health.model_version}
            </span>
          </>
        )}
        <button className="icon-btn" onClick={onRefresh} title="刷新状态">
          <RefreshCw size={18} />
        </button>
      </div>
    </header>
  );
}
