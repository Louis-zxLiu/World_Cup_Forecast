import React, { useEffect, useState } from "react";
import { Calendar, Play, RefreshCw } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  api,
  type GroupStandingRecord,
  type TournamentBracket,
  type TournamentFixture,
  type TournamentScheduleImport,
} from "../api";

const DEFAULT_TEAMS = `Brazil
Argentina
France
England
Spain
Germany
Portugal
Netherlands
Belgium
Croatia
Uruguay
Colombia
United States
Mexico
Japan
Morocco
Senegal
Australia
South Korea
Denmark
Switzerland
Poland
Saudi Arabia
Ecuador`;

export function Tournament() {
  const [teamsText, setTeamsText] = useState(DEFAULT_TEAMS);
  const [simulations, setSimulations] = useState(5000);
  const [loading, setLoading] = useState(false);
  const [standingsLoading, setStandingsLoading] = useState(false);
  const [importLoading, setImportLoading] = useState(false);
  const [useCurrentState, setUseCurrentState] = useState(true);
  const [result, setResult] = useState<TournamentBracket | null>(null);
  const [standings, setStandings] = useState<GroupStandingRecord[]>([]);
  const [fixtures, setFixtures] = useState<TournamentFixture[]>([]);
  const [importedAt, setImportedAt] = useState("");
  const [message, setMessage] = useState("");

  async function loadStandings() {
    try {
      const rows = await api<GroupStandingRecord[]>("/api/standings/latest");
      setStandings(rows);
    } catch {
      setStandings([]);
    }
  }

  useEffect(() => {
    loadStandings();
  }, []);

  async function syncStandings() {
    setStandingsLoading(true);
    setMessage("");
    try {
      const result = await api<{ standing_records: number }>("/api/standings/sync", {
        method: "POST",
      });
      await loadStandings();
      setMessage(`积分榜同步完成：${result.standing_records} 条记录。`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "积分榜同步失败");
    } finally {
      setStandingsLoading(false);
    }
  }

  async function importSchedule() {
    setImportLoading(true);
    setMessage("");
    try {
      const data = await api<TournamentScheduleImport>("/api/tournament/schedule/import", {
        method: "POST",
      });
      setFixtures(data.fixtures);
      setStandings(data.standings);
      setImportedAt(data.imported_at);
      if (data.teams.length >= 2) {
        setTeamsText(data.teams.join("\n"));
      }
      setUseCurrentState(true);
      setMessage(`赛程导入完成：${data.fixtures.length} 场比赛，${data.teams.length} 支球队。`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "赛程导入失败");
    } finally {
      setImportLoading(false);
    }
  }

  async function simulate() {
    const teams = teamsText.split("\n").map((team) => team.trim()).filter(Boolean);
    if (teams.length < 2) {
      setMessage("请至少输入 2 支球队。");
      return;
    }
    setLoading(true);
    setMessage("");
    try {
      const data = await api<TournamentBracket>("/api/predict/tournament", {
        method: "POST",
        body: JSON.stringify({
          teams,
          simulations,
          fixtures,
          standings,
          use_current_state: useCurrentState,
        }),
      });
      setResult(data);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "模拟失败");
    } finally {
      setLoading(false);
    }
  }

  const chartData = result?.results.slice(0, 16).map((row) => ({
    name: row.team.length > 12 ? `${row.team.slice(0, 12)}...` : row.team,
    fullName: row.team,
    champion: +(row.champion * 100).toFixed(1),
  })) ?? [];
  const completedFixtures = fixtures.filter((fixture) => fixture.completed).length;
  const remainingFixtures = fixtures.length - completedFixtures;

  return (
    <div className="stack">
      <section className="card info-card">
        <div className="page-heading compact">
          <div>
            <h2>小组赛出线形势</h2>
            <p>
              小组赛预测会考虑积分、净胜球、剩余赛程和战意。这里先同步积分榜数据，
              后续会把“必须赢/可接受平局/已出线轮换”等形势转成模型特征。
            </p>
          </div>
          <button className="btn btn-primary" onClick={syncStandings} disabled={standingsLoading}>
            <RefreshCw size={18} />
            {standingsLoading ? "同步中..." : "同步积分榜"}
          </button>
        </div>
        {standings.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>小组</th>
                  <th>排名</th>
                  <th>球队</th>
                  <th>赛</th>
                  <th>胜</th>
                  <th>平</th>
                  <th>负</th>
                  <th>进/失</th>
                  <th>净胜球</th>
                  <th>积分</th>
                </tr>
              </thead>
              <tbody>
                {standings.map((row) => (
                  <tr key={`${row.group_name}-${row.rank}-${row.team}`}>
                    <td>{row.group_name}</td>
                    <td>{row.rank}</td>
                    <td>
                      <strong>{row.team}</strong>
                    </td>
                    <td>{row.played}</td>
                    <td>{row.wins}</td>
                    <td>{row.draws}</td>
                    <td>{row.losses}</td>
                    <td>
                      {row.goals_for}/{row.goals_against}
                    </td>
                    <td>{row.goal_difference}</td>
                    <td className="odds-win">{row.points}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty compact">暂无积分榜数据。点击“同步积分榜”抓取公开积分榜。</div>
        )}
      </section>

      <div className="grid-2">
        <section className="card">
          <div className="page-heading compact">
            <div>
              <h2>Monte Carlo 锦标赛模拟</h2>
              <p>可手工输入球队，也可以一键导入公开赛程；导入后支持从当前积分和已赛结果继续推演。</p>
            </div>
          </div>
          <div className="form-stack">
            <div className="btn-row">
              <button className="btn btn-primary" onClick={importSchedule} disabled={importLoading}>
                <Calendar size={18} />
                {importLoading ? "导入中..." : "一键导入赛程"}
              </button>
              <button className="btn btn-ghost" onClick={syncStandings} disabled={standingsLoading}>
                <RefreshCw size={18} />
                {standingsLoading ? "同步中..." : "只同步积分榜"}
              </button>
            </div>
            <div className="formula-box">
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={useCurrentState}
                  onChange={(event) => setUseCurrentState(event.target.checked)}
                />
                按当前结果继续推演
              </label>
              <span className="muted">
                开启后使用当前积分榜作为小组赛起点，只模拟剩余未完成比赛。
              </span>
            </div>
            {fixtures.length > 0 && (
              <div className="info-card">
                <strong>已导入赛程</strong>
                <p>
                  共 {fixtures.length} 场，已完成 {completedFixtures} 场，待模拟 {remainingFixtures} 场。
                  {importedAt ? ` 导入时间：${new Date(importedAt).toLocaleString("zh-CN")}` : ""}
                </p>
              </div>
            )}
            <label>
              参赛球队
              <textarea rows={14} value={teamsText} onChange={(event) => setTeamsText(event.target.value)} />
            </label>
            <label>
              模拟次数
              <input
                type="number"
                min={100}
                max={100000}
                value={simulations}
                onChange={(event) => setSimulations(Number(event.target.value))}
              />
            </label>
            {message && <div className="inline-error">{message}</div>}
            <button className="btn btn-primary btn-large" onClick={simulate} disabled={loading}>
              <Play size={18} />
              {loading ? "模拟中..." : "开始模拟"}
            </button>
          </div>
        </section>

        <section className="card">
          <div className="page-heading compact">
            <div>
              <h2>夺冠概率 Top 16</h2>
              <p>
                用于快速观察整体强弱和路径概率。
                {result ? ` 已使用 ${result.fixtures_used} 场赛程，当前状态：${result.current_state_used ? "是" : "否"}。` : ""}
              </p>
            </div>
          </div>
          {chartData.length > 0 ? (
            <div className="chart-wrap tournament">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 24 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" unit="%" />
                  <YAxis type="category" dataKey="name" width={120} />
                  <Tooltip formatter={(value, _name, payload) => [`${value}%`, payload.payload.fullName]} />
                  <Bar dataKey="champion" fill="#216869" radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="empty">点击“开始模拟”后显示结果。</div>
          )}
        </section>
      </div>

      {fixtures.length > 0 && (
        <section className="card">
          <div className="page-heading compact">
            <div>
              <h2>赛程预览</h2>
              <p>小组字段会用于出线形势推演；没有小组的比赛会作为淘汰赛或待识别赛程保留。</p>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>时间</th>
                  <th>阶段</th>
                  <th>小组</th>
                  <th>比赛</th>
                  <th>比分/状态</th>
                </tr>
              </thead>
              <tbody>
                {fixtures.slice(0, 80).map((fixture) => (
                  <tr key={fixture.match_id}>
                    <td>
                      {fixture.kickoff_time ? new Date(fixture.kickoff_time).toLocaleString("zh-CN") : "-"}
                    </td>
                    <td>{fixture.round_name}</td>
                    <td>{fixture.group_name ?? "待识别"}</td>
                    <td>
                      <strong>{fixture.home_team}</strong>
                      <span className="muted"> vs </span>
                      <strong>{fixture.away_team}</strong>
                    </td>
                    <td>
                      {fixture.completed
                        ? `${fixture.home_score ?? 0}-${fixture.away_score ?? 0}`
                        : "未赛，将模拟"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {result && (
        <section className="card">
          <div className="page-heading compact">
            <div>
              <h2>完整晋级概率表</h2>
              <p>{result.simulations.toLocaleString()} 次模拟</p>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>球队</th>
                  <th>小组出线</th>
                  <th>八强</th>
                  <th>四强</th>
                  <th>决赛</th>
                  <th>夺冠</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((row, index) => (
                  <tr key={row.team}>
                    <td>{index + 1}</td>
                    <td>
                      <strong>{row.team}</strong>
                    </td>
                    <td>{(row.group_advance * 100).toFixed(1)}%</td>
                    <td>{(row.quarter_final * 100).toFixed(1)}%</td>
                    <td>{(row.semi_final * 100).toFixed(1)}%</td>
                    <td>{(row.final * 100).toFixed(1)}%</td>
                    <td className="odds-win">{(row.champion * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
