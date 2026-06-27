import React, { useMemo, useState } from "react";
import { Download, Play } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type BacktestRunResult, type WorldCupDataImportResult } from "../api";
import { MarkdownView } from "./MarkdownView";

export function Backtest() {
  const [loading, setLoading] = useState(false);
  const [ingestLoading, setIngestLoading] = useState(false);
  const [result, setResult] = useState<BacktestRunResult | null>(null);
  const [dataImport, setDataImport] = useState<WorldCupDataImportResult | null>(null);
  const [message, setMessage] = useState("");
  const [edgeThreshold, setEdgeThreshold] = useState(0.025);
  const [kellyFraction, setKellyFraction] = useState(0.25);
  const [maxStake, setMaxStake] = useState(0.05);

  async function ingestData() {
    setIngestLoading(true);
    setMessage("");
    try {
      const data = await api<WorldCupDataImportResult>("/api/ingest/run?force_download=true", {
        method: "POST",
      });
      setDataImport(data);
      setMessage(`历史数据已整理：${data.imported_records} 场，覆盖 ${data.years[0]}-${data.years[data.years.length - 1]}，Elo 球队 ${data.teams_with_elo} 支。`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "历史数据下载整理失败");
    } finally {
      setIngestLoading(false);
    }
  }

  async function runBacktest() {
    setLoading(true);
    setMessage("");
    try {
      const data = await api<BacktestRunResult>("/api/backtest/run", {
        method: "POST",
        body: JSON.stringify({
          years: [2014, 2018, 2022],
          edge_threshold: edgeThreshold,
          kelly_fraction: kellyFraction,
          max_stake_fraction: maxStake,
          initial_bankroll: 1000,
          home_odds: 2.5,
          draw_odds: 3.1,
          away_odds: 2.5,
        }),
      });
      setResult(data);
      if (data.metrics.length === 0) {
        setMessage("后端已执行回测，但没有找到 2014/2018/2022 的历史比赛记录。请先确认 data/wc_results.csv 已存在。");
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "回测失败");
    } finally {
      setLoading(false);
    }
  }

  const chartData = useMemo(
    () =>
      result?.metrics.map((row) => ({
        year: String(row.tournament_year),
        "模型 Brier": +row.brier.toFixed(4),
        "基准 Brier": +row.baseline_brier.toFixed(4),
        "ROI %": +(row.roi * 100).toFixed(2),
      })) ?? [],
    [result],
  );

  return (
    <div className="stack">
      <section className="card">
        <div className="page-heading">
          <div>
            <h2>参数回测研究</h2>
            <p>像量化交易一样记录参数、跑历史表现、比较 ROI、Brier、LogLoss 和最大回撤；首次运行会自动导入内置历史世界杯数据。</p>
          </div>
          <div className="btn-row">
            <button className="btn btn-ghost" onClick={ingestData} disabled={ingestLoading}>
              <Download size={18} />
              {ingestLoading ? "整理中..." : "联网整理历史数据"}
            </button>
            <button className="btn btn-primary" onClick={runBacktest} disabled={loading}>
              <Play size={18} />
              {loading ? "回测中..." : "运行历史回测"}
            </button>
          </div>
        </div>

        <div className="param-grid">
          <label>
            Edge 入场阈值
            <input
              type="number"
              step="0.005"
              value={edgeThreshold}
              onChange={(event) => setEdgeThreshold(Number(event.target.value))}
            />
          </label>
          <label>
            Kelly 折扣
            <input
              type="number"
              step="0.05"
              value={kellyFraction}
              onChange={(event) => setKellyFraction(Number(event.target.value))}
            />
          </label>
          <label>
            单项最大仓位
            <input
              type="number"
              step="0.01"
              value={maxStake}
              onChange={(event) => setMaxStake(Number(event.target.value))}
            />
          </label>
        </div>
        <div className="formula-box">
          <strong>当前研究组合：</strong>
          <span>
            Edge 阈值 {edgeThreshold}，Kelly 折扣 {kellyFraction}，单项最大仓位 {maxStake}。
            这些参数会直接传入后端回测 Runner，影响模拟下注仓位和 ROI。
          </span>
        </div>
        {dataImport && (
          <div className="info-card">
            <strong>历史数据状态：</strong>
            <span>
              {dataImport.message} 已导入 {dataImport.imported_records} 场，标准文件：
              {dataImport.output_path}
            </span>
          </div>
        )}
        {message && <div className="inline-error">{message}</div>}
      </section>

      {result && (
        <>
          {result.explanation && (
            <section className="card">
              <div className="backtest-explanation">
                <MarkdownView content={result.explanation} />
              </div>
            </section>
          )}
          <section className="card">
            <div className="page-heading compact">
              <div>
                <h2>回测指标汇总</h2>
                <p>越低的 Brier/LogLoss 表示概率校准越好；ROI 和最大回撤用于评估盈利质量。</p>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>年份</th>
                    <th>场数</th>
                    <th>Brier</th>
                    <th>基准 Brier</th>
                    <th>LogLoss</th>
                    <th>基准 LogLoss</th>
                    <th>ROI</th>
                    <th>最大回撤</th>
                  </tr>
                </thead>
                <tbody>
                  {result.metrics.map((row) => (
                    <tr key={row.tournament_year}>
                      <td>
                        <strong>{row.tournament_year}</strong>
                      </td>
                      <td>{row.record_count}</td>
                      <td>{row.brier.toFixed(4)}</td>
                      <td>{row.baseline_brier.toFixed(4)}</td>
                      <td>{row.log_loss.toFixed(4)}</td>
                      <td>{row.baseline_log_loss.toFixed(4)}</td>
                      <td className={row.roi >= 0 ? "edge-positive" : "edge-negative"}>
                        {(row.roi * 100).toFixed(2)}%
                      </td>
                      <td>{(row.max_drawdown * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <div className="grid-2">
            <section className="card">
              <h2>Brier Score 对比</h2>
              <div className="chart-wrap backtest">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis domain={["auto", "auto"]} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="模型 Brier" stroke="#216869" strokeWidth={3} />
                    <Line type="monotone" dataKey="基准 Brier" stroke="#b7791f" strokeWidth={3} strokeDasharray="5 5" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="card">
              <h2>模拟投注 ROI</h2>
              <div className="chart-wrap backtest">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="year" />
                    <YAxis unit="%" />
                    <Tooltip formatter={(value) => `${value}%`} />
                    <Legend />
                    <Line type="monotone" dataKey="ROI %" stroke="#b83255" strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
