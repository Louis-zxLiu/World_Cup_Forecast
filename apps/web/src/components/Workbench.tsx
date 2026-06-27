import React, { useMemo, useState } from "react";
import { Activity, BarChart3, Calculator, RefreshCw, SlidersHorizontal, Target } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, streamPrediction, type AgentReasoning, type LiveMatch, type Prediction } from "../api";
import { MarkdownView } from "./MarkdownView";
import { ReasoningTrace } from "./ReasoningTrace";

type Props = {
  liveMatches: LiveMatch[];
  onRefreshMatches: () => void;
  onMessage: (message: string, type?: "info" | "error") => void;
};

type ResearchParams = {
  kelly_fraction: number;
  max_stake_fraction: number;
  value_edge_threshold: number;
  home_advantage_elo: number;
  draw_bias: number;
  goal_rate_multiplier: number;
};

const OUTCOME_LABELS: Record<string, string> = {
  home_win: "主胜",
  draw: "平局",
  away_win: "客胜",
};

const OUTCOME_COLORS: Record<string, string> = {
  home_win: "#216869",
  draw: "#b7791f",
  away_win: "#b83255",
};

const DEFAULT_PARAMS: ResearchParams = {
  kelly_fraction: 0.25,
  max_stake_fraction: 0.05,
  value_edge_threshold: 0.025,
  home_advantage_elo: 45,
  draw_bias: 1,
  goal_rate_multiplier: 1,
};

function pct(value?: number | null) {
  if (value == null) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function outcomeName(key: string) {
  return OUTCOME_LABELS[key] ?? key;
}

function MatchCard({ match, active, onClick }: { match: LiveMatch; active: boolean; onClick: () => void }) {
  const hasScore = match.home_score != null && match.away_score != null;
  const isLive = match.status_state === "in";
  const time = new Date(match.date).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <button className={`match-card${active ? " active" : ""}${isLive ? " live" : ""}`} onClick={onClick}>
      <span>
        <strong>{match.home_team}</strong>
        <small>{match.home_abbr || "主队"}</small>
      </span>
      <span className="match-center">
        {hasScore ? <strong>{match.home_score} - {match.away_score}</strong> : <strong>VS</strong>}
        <small>{isLive ? `进行中 ${match.display_clock}` : time}</small>
      </span>
      <span>
        <strong>{match.away_team}</strong>
        <small>{match.away_abbr || "客队"}</small>
      </span>
    </button>
  );
}

export function Workbench({ liveMatches, onRefreshMatches, onMessage }: Props) {
  const [selectedMatch, setSelectedMatch] = useState<LiveMatch | null>(null);
  const [homeTeam, setHomeTeam] = useState("Brazil");
  const [awayTeam, setAwayTeam] = useState("Germany");
  const [bankroll, setBankroll] = useState(10000);
  const [neutralSite, setNeutralSite] = useState(true);
  const [params, setParams] = useState<ResearchParams>(DEFAULT_PARAMS);
  const [loading, setLoading] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [reasonings, setReasonings] = useState<AgentReasoning[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [detailTab, setDetailTab] = useState<"reasoning" | "signals" | "agents" | "features">("reasoning");

  const home = selectedMatch?.home_team || homeTeam;
  const away = selectedMatch?.away_team || awayTeam;

  const probabilityRows = useMemo(() => {
    if (!prediction) return [];
    return (["home_win", "draw", "away_win"] as const).map((key) => ({
      key,
      name: OUTCOME_LABELS[key],
      value: prediction.probabilities[key],
      color: OUTCOME_COLORS[key],
    }));
  }, [prediction]);

  const bestOutcome = probabilityRows.reduce(
    (best, row) => (row.value > best.value ? row : best),
    { key: "", name: "", value: -1, color: "" },
  );

  async function syncESPN() {
    setLoading("espn");
    try {
      const result = await api<{ matches_synced: number; teams_synced: number }>("/api/espn/sync", {
        method: "POST",
      });
      onRefreshMatches();
      onMessage(`赛程同步完成：${result.matches_synced} 场比赛，${result.teams_synced} 支球队。`, "info");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "赛程同步失败", "error");
    } finally {
      setLoading("");
    }
  }

  async function scrapeOdds() {
    setLoading("odds");
    try {
      const result = await api<{ record_count: number; warnings: string[] }>(
        "/api/odds/china-lottery/scrape",
        { method: "POST" },
      );
      onMessage(`500彩票网赔率抓取完成：${result.record_count} 条。${result.warnings.join(" ")}`, "info");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "赔率抓取失败", "error");
    } finally {
      setLoading("");
    }
  }

  async function predict() {
    if (!home || !away) {
      onMessage("请先选择比赛，或手动输入主队和客队。", "error");
      return;
    }
    setLoading("predict");
    setStreaming(true);
    setReasonings([]);
    setPrediction(null);
    setDetailTab("reasoning");
    const findings: AgentReasoning[] = [];
    try {
      await streamPrediction(
        {
          home_team: home,
          away_team: away,
          neutral_site: neutralSite,
          bankroll,
          ...params,
        },
        {
          onPrediction: (data) => {
            setPrediction({
              probabilities: data.probabilities,
              expected_score: data.expected_score,
              most_likely_score: data.most_likely_score,
              odds: data.odds ?? undefined,
              bet_signals: data.bet_signals ?? [],
              agent_findings: [],
              explanation: "",
              shap_values: data.shap_values ?? {},
            });
          },
          onReasoning: (data) => {
            findings.push(data);
            setReasonings([...findings]);
          },
          onReport: (data) => {
            setPrediction((prev) =>
              prev
                ? {
                    ...prev,
                    explanation: data.explanation,
                    agent_findings: findings.map((f) => ({
                      agent: f.agent,
                      confidence: f.confidence,
                      signal: f.signal,
                      rationale: f.rationale,
                      sources: f.sources,
                      metrics: f.metrics,
                    })),
                  }
                : prev,
            );
          },
        },
      );
      onMessage("预测完成。可在“推理过程”查看每位智能体的逐步推理。", "info");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "预测失败", "error");
    } finally {
      setStreaming(false);
      setLoading("");
    }
  }

  function selectMatch(match: LiveMatch) {
    setSelectedMatch(match);
    setHomeTeam(match.home_team);
    setAwayTeam(match.away_team);
    setPrediction(null);
  }

  function updateParam(key: keyof ResearchParams, value: number) {
    setParams((current) => ({ ...current, [key]: value }));
  }

  const shapRows = prediction?.shap_values
    ? Object.entries(prediction.shap_values)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .map(([name, value]) => ({ name, value, color: value >= 0 ? "#216869" : "#b83255" }))
    : [];

  return (
    <div className="workbench">
      <section className="guide-strip">
        <div>
          <strong>1. 选择比赛</strong>
          <span>同步赛程或手动输入球队</span>
        </div>
        <div>
          <strong>2. 调整参数</strong>
          <span>像量化交易一样测试公式</span>
        </div>
        <div>
          <strong>3. 运行预测</strong>
          <span>查看概率、Edge 和仓位</span>
        </div>
      </section>

      <div className="grid-sidebar wide">
        <aside className="stack">
          <section className="card">
            <div className="card-title">
              <Target size={20} />
              比赛选择
            </div>
            <div className="btn-row">
              <button className="btn btn-ghost" onClick={syncESPN} disabled={loading === "espn"}>
                <RefreshCw size={18} />
                同步赛程
              </button>
              <button className="btn btn-ghost" onClick={scrapeOdds} disabled={loading === "odds"}>
                <RefreshCw size={18} />
                抓取500赔率
              </button>
            </div>

            <div className="match-list">
              {liveMatches.length === 0 ? (
                <div className="empty compact">暂无赛程。可以点击“同步赛程”，或直接手动输入球队。</div>
              ) : (
                liveMatches.slice(0, 12).map((match) => (
                  <MatchCard
                    key={match.match_id}
                    match={match}
                    active={selectedMatch?.match_id === match.match_id}
                    onClick={() => selectMatch(match)}
                  />
                ))
              )}
            </div>
          </section>

          <section className="card">
            <div className="card-title">
              <Calculator size={20} />
              手动输入
            </div>
            <label>
              主队
              <input value={homeTeam} onChange={(event) => setHomeTeam(event.target.value)} />
            </label>
            <label>
              客队
              <input value={awayTeam} onChange={(event) => setAwayTeam(event.target.value)} />
            </label>
            <label>
              本金
              <input type="number" value={bankroll} onChange={(event) => setBankroll(Number(event.target.value))} />
            </label>
            <label className="checkbox-row">
              <input type="checkbox" checked={neutralSite} onChange={(event) => setNeutralSite(event.target.checked)} />
              中立场比赛
            </label>
          </section>
        </aside>

        <section className="stack">
          <section className="card research-card">
            <div className="card-title">
              <SlidersHorizontal size={20} />
              研究参数与公式
            </div>
            <div className="param-grid">
              <label>
                Kelly 折扣
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  value={params.kelly_fraction}
                  onChange={(event) => updateParam("kelly_fraction", Number(event.target.value))}
                />
                <small>0.25 表示四分之一 Kelly，越小越保守。</small>
              </label>
              <label>
                单项最大仓位
                <input
                  type="number"
                  min="0"
                  max="0.5"
                  step="0.01"
                  value={params.max_stake_fraction}
                  onChange={(event) => updateParam("max_stake_fraction", Number(event.target.value))}
                />
                <small>0.05 表示单个结果最多押 5% 本金。</small>
              </label>
              <label>
                Edge 入场阈值
                <input
                  type="number"
                  min="-0.2"
                  max="0.5"
                  step="0.005"
                  value={params.value_edge_threshold}
                  onChange={(event) => updateParam("value_edge_threshold", Number(event.target.value))}
                />
                <small>模型概率减市场隐含概率，高于该值才视为价值。</small>
              </label>
              <label>
                主场 Elo 加成
                <input
                  type="number"
                  min="0"
                  max="200"
                  step="5"
                  value={params.home_advantage_elo}
                  onChange={(event) => updateParam("home_advantage_elo", Number(event.target.value))}
                />
                <small>非中立场时生效，用于研究主场优势。</small>
              </label>
              <label>
                平局偏置
                <input
                  type="number"
                  min="0.5"
                  max="1.8"
                  step="0.05"
                  value={params.draw_bias}
                  onChange={(event) => updateParam("draw_bias", Number(event.target.value))}
                />
                <small>提高或降低平局概率，适合研究淘汰赛/小组赛差异。</small>
              </label>
              <label>
                进球倍率
                <input
                  type="number"
                  min="0.5"
                  max="1.8"
                  step="0.05"
                  value={params.goal_rate_multiplier}
                  onChange={(event) => updateParam("goal_rate_multiplier", Number(event.target.value))}
                />
                <small>放大或压低双方期望进球，用于研究大/小球环境。</small>
              </label>
            </div>
            <div className="formula-box">
              <strong>当前公式：</strong>
              <span>
                Edge = 模型概率 - 赔率去水隐含概率；仓位 = min(最大仓位, Kelly × 折扣) × 本金。
              </span>
            </div>
            <button className="btn btn-primary btn-large" onClick={predict} disabled={loading === "predict"}>
              <Activity size={20} />
              {loading === "predict" ? "正在计算..." : "开始预测"}
            </button>
          </section>

          <section className="card result-card">
            <div className="card-title">
              <BarChart3 size={20} />
              预测结果
            </div>
            {!prediction ? (
              <div className="empty">完成上方三步后，这里会显示胜平负概率、比分、赔率 Edge 和建议仓位。</div>
            ) : (
              <>
                <div className="result-header">
                  <div>
                    <h2>
                      {home} <span>vs</span> {away}
                    </h2>
                    <p>
                      期望比分 {prediction.expected_score[0].toFixed(2)} -{" "}
                      {prediction.expected_score[1].toFixed(2)}，最可能比分 {prediction.most_likely_score}
                    </p>
                  </div>
                  <div className="best-chip">最高概率：{bestOutcome.name}</div>
                </div>

                <div className="prob-row">
                  {probabilityRows.map((row) => (
                    <div className={`prob-tile${row.key === bestOutcome.key ? " best" : ""}`} key={row.key}>
                      <span>{row.name}</span>
                      <strong>{pct(row.value)}</strong>
                    </div>
                  ))}
                </div>

                <div className="chart-wrap large">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={probabilityRows}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis tickFormatter={(value) => `${Number(value) * 100}%`} />
                      <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`} />
                      <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                        {probabilityRows.map((row) => (
                          <Cell key={row.key} fill={row.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="explanation-box">
                  <MarkdownView content={prediction.explanation} />
                </div>
              </>
            )}
          </section>
        </section>
      </div>

      {prediction && (
        <section className="card details-card">
          <div className="segmented">
            <button className={detailTab === "reasoning" ? "active" : ""} onClick={() => setDetailTab("reasoning")}>
              推理过程
            </button>
            <button className={detailTab === "signals" ? "active" : ""} onClick={() => setDetailTab("signals")}>
              投注信号
            </button>
            <button className={detailTab === "agents" ? "active" : ""} onClick={() => setDetailTab("agents")}>
              多智能体解释
            </button>
            <button className={detailTab === "features" ? "active" : ""} onClick={() => setDetailTab("features")}>
              特征贡献
            </button>
          </div>

          {detailTab === "reasoning" && (
            <ReasoningTrace reasonings={reasonings} streaming={streaming} />
          )}

          {detailTab === "signals" && (
            <div className="signal-grid">
              {prediction.bet_signals.map((signal) => (
                <article className="signal-card" key={signal.outcome}>
                  <h3>{outcomeName(signal.outcome)}</h3>
                  <dl>
                    <div>
                      <dt>模型概率</dt>
                      <dd>{pct(signal.model_probability)}</dd>
                    </div>
                    <div>
                      <dt>市场概率</dt>
                      <dd>{pct(signal.market_probability)}</dd>
                    </div>
                    <div>
                      <dt>Edge</dt>
                      <dd className={(signal.edge ?? 0) >= 0 ? "edge-positive" : "edge-negative"}>{pct(signal.edge)}</dd>
                    </div>
                    <div>
                      <dt>建议仓位</dt>
                      <dd>{signal.stake.toFixed(2)}</dd>
                    </div>
                  </dl>
                  <p>{signal.rationale}</p>
                </article>
              ))}
            </div>
          )}

          {detailTab === "agents" && (
            <div className="agent-list">
              {prediction.agent_findings.map((finding, index) => (
                <article className={`agent-card ${finding.signal}`} key={`${finding.agent}-${index}`}>
                  <div>
                    <strong>{finding.agent}</strong>
                    <span>置信度 {(finding.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <p>{finding.rationale}</p>
                </article>
              ))}
            </div>
          )}

          {detailTab === "features" && (
            shapRows.length > 0 ? (
              <div className="chart-wrap feature">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={shapRows} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="name" width={160} />
                    <Tooltip formatter={(value) => Number(value).toFixed(4)} />
                    <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                      {shapRows.map((row) => (
                        <Cell key={row.name} fill={row.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="empty">训练 XGBoost 模型后，这里会显示 SHAP 特征贡献。</div>
            )
          )}
        </section>
      )}
    </div>
  );
}
