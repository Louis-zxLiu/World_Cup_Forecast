import React, { useState } from "react";
import { RefreshCw } from "lucide-react";
import { api, type OddsRecord } from "../api";

type Props = {
  odds: OddsRecord[];
  onRefresh: (odds: OddsRecord[]) => void;
  onMessage: (message: string, type?: "info" | "error") => void;
};

export function OddsLog({ odds, onRefresh, onMessage }: Props) {
  const [loading, setLoading] = useState(false);

  async function scrape() {
    setLoading(true);
    try {
      const result = await api<{ record_count: number; warnings: string[] }>(
        "/api/odds/china-lottery/scrape",
        { method: "POST" },
      );
      const fresh = await api<OddsRecord[]>("/api/odds/latest?limit=200");
      onRefresh(fresh);
      onMessage(`抓取完成：${result.record_count} 条。${result.warnings.join(" ")}`, "info");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "抓取失败", "error");
    } finally {
      setLoading(false);
    }
  }

  const lastUpdate = odds.length > 0 ? new Date(odds[0].scraped_at).toLocaleString("zh-CN") : "暂无";
  const cleaningLabel = (method?: string) => {
    if (method === "llm") return "LLM清洗";
    if (method === "rule_fallback") return "规则降级";
    return "规则清洗";
  };

  return (
    <section className="card">
      <div className="page-heading">
        <div>
          <h2>500彩票网赔率</h2>
          <p>用于去水、计算市场隐含概率和价值投注 Edge。最后更新：{lastUpdate}</p>
        </div>
        <button className="btn btn-primary" onClick={scrape} disabled={loading}>
          <RefreshCw size={18} />
          {loading ? "抓取中..." : "立即抓取"}
        </button>
      </div>

      {odds.length === 0 ? (
        <div className="empty">暂无赔率数据。点击“立即抓取”后，系统会保存原始快照并解析胜平负赔率。</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>编号</th>
                <th>比赛</th>
                <th>玩法</th>
                <th>盘口</th>
                <th>胜</th>
                <th>平</th>
                <th>负</th>
                <th>清洗</th>
                <th>更新时间</th>
              </tr>
            </thead>
            <tbody>
              {odds.map((row) => (
                <tr key={`${row.match_id}-${row.play_type}-${row.handicap ?? ""}`}>
                  <td className="mono">{row.match_id}</td>
                  <td>
                    <strong>{row.home_team}</strong>
                    <span className="muted"> vs </span>
                    <strong>{row.away_team}</strong>
                  </td>
                  <td>{row.play_type}</td>
                  <td>{row.handicap ?? "-"}</td>
                  <td className="odds-win">{row.win_odds ?? "-"}</td>
                  <td className="odds-draw">{row.draw_odds ?? "-"}</td>
                  <td className="odds-lose">{row.lose_odds ?? "-"}</td>
                  <td>
                    <span className="pill">{cleaningLabel(row.raw?.cleaning?.method)}</span>
                  </td>
                  <td>{new Date(row.scraped_at).toLocaleString("zh-CN")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
