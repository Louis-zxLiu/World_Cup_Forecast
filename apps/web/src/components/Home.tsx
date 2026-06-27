import React, { useState } from "react";
import { Sparkles, TrendingUp, Search, Loader2 } from "lucide-react";
import { api, type AskResponse, type TodayCard, type TodayOverview } from "../api";

const OUTCOME_LABELS: Record<string, string> = {
  home_win: "主胜",
  draw: "平局",
  away_win: "客胜",
};

function pct(value?: number | null) {
  if (value == null) return "-";
  return `${(value * 100).toFixed(0)}%`;
}

function TodayCardView({ card }: { card: TodayCard }) {
  const probs = card.probabilities;
  const best = (["home_win", "draw", "away_win"] as const).reduce(
    (acc, key) => (probs[key] > acc.value ? { key, value: probs[key] } : acc),
    { key: "home_win", value: -1 },
  );
  return (
    <article className={`today-card${card.has_value ? " has-value" : ""}`}>
      <div className="today-card-head">
        <strong>
          {card.home_team} <span>vs</span> {card.away_team}
        </strong>
        {card.has_value && <span className="value-flag">价值机会</span>}
      </div>
      <div className="today-probs">
        {(["home_win", "draw", "away_win"] as const).map((key) => (
          <div key={key} className={`today-prob${key === best.key ? " best" : ""}`}>
            <small>{OUTCOME_LABELS[key]}</small>
            <strong>{pct(probs[key])}</strong>
          </div>
        ))}
      </div>
      <div className="today-meta">
        <span>最可能比分 {card.most_likely_score}</span>
        {card.recommendation && (
          <span className="today-rec">
            建议：{OUTCOME_LABELS[card.recommendation.outcome]} · edge{" "}
            {((card.recommendation.edge ?? 0) * 100).toFixed(1)}% · 仓位{" "}
            {card.recommendation.stake.toFixed(0)}
          </span>
        )}
      </div>
    </article>
  );
}

export function Home({ onMessage }: { onMessage: (m: string, type?: "info" | "error") => void }) {
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [overview, setOverview] = useState<TodayOverview | null>(null);
  const [loadingToday, setLoadingToday] = useState(false);

  async function ask() {
    if (!question.trim()) return;
    setAsking(true);
    setAnswer(null);
    try {
      const result = await api<AskResponse>("/api/ask", {
        method: "POST",
        body: JSON.stringify({ question }),
      });
      setAnswer(result);
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "提问失败", "error");
    } finally {
      setAsking(false);
    }
  }

  async function loadToday() {
    setLoadingToday(true);
    try {
      const result = await api<TodayOverview>("/api/predict/today");
      setOverview(result);
      if (result.count === 0) {
        onMessage("暂无赛程。可在“预测工作台”同步赛程后再试。", "info");
      }
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "加载今日推荐失败", "error");
    } finally {
      setLoadingToday(false);
    }
  }

  return (
    <div className="home">
      <section className="card ask-card">
        <div className="card-title">
          <Sparkles size={20} />
          直接提问
        </div>
        <div className="ask-row">
          <Search size={18} />
          <input
            placeholder="例如：巴西对德国谁会赢？今晚有什么值得买的？"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && ask()}
          />
          <button className="btn btn-primary" onClick={ask} disabled={asking}>
            {asking ? <Loader2 size={18} className="spin" /> : "提问"}
          </button>
        </div>
        {answer && (
          <div className={`ask-answer${answer.matched ? "" : " unmatched"}`}>
            {answer.answer}
          </div>
        )}
      </section>

      <section className="card">
        <div className="card-title">
          <TrendingUp size={20} />
          今日推荐
          <button className="btn btn-ghost today-refresh" onClick={loadToday} disabled={loadingToday}>
            {loadingToday ? <Loader2 size={16} className="spin" /> : "一键生成"}
          </button>
        </div>
        {!overview ? (
          <div className="empty">点击「一键生成」，自动为赛程上的每场比赛出预测和价值推荐，无需手动输入。</div>
        ) : overview.count === 0 ? (
          <div className="empty">暂无可预测的赛程。</div>
        ) : (
          <div className="today-grid">
            {overview.cards.map((card) => (
              <TodayCardView key={card.match_id} card={card} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
