import React, { useEffect, useState } from "react";
import { Activity, Database, Globe, Save } from "lucide-react";
import {
  api,
  type LLMSettings,
  type PublicLLMSettings,
  type PublicSearchSettings,
  type SearchSettings,
  type WorldCupDataImportResult,
} from "../api";

type Props = {
  settings: LLMSettings;
  apiKeySaved: boolean;
  onChange: (settings: LLMSettings) => void;
  onMessage: (message: string, type?: "info" | "error") => void;
  onSaved: () => void;
};

const DEFAULT_SEARCH: SearchSettings = {
  provider: "bocha",
  base_url: "https://api.bochaai.com/v1/web-search",
  api_key: "",
  timeout_seconds: 15,
  max_results: 6,
  enabled: false,
};

const SEARCH_PRESETS: Record<string, string> = {
  bocha: "https://api.bochaai.com/v1/web-search",
  zhipu: "https://open.bigmodel.cn/api/paas/v4/web_search",
  custom: "",
  none: "",
};

export function Settings({ settings, apiKeySaved, onChange, onMessage, onSaved }: Props) {
  const [loading, setLoading] = useState("");
  const [search, setSearch] = useState<SearchSettings>(DEFAULT_SEARCH);
  const [searchKeySaved, setSearchKeySaved] = useState(false);

  useEffect(() => {
    api<PublicSearchSettings>("/api/settings/search")
      .then((s) => {
        setSearch({ ...DEFAULT_SEARCH, ...s, api_key: "" });
        setSearchKeySaved(s.api_key_saved);
      })
      .catch(() => {
        /* keep defaults if backend unreachable */
      });
  }, []);

  function set(patch: Partial<LLMSettings>) {
    onChange({ ...settings, ...patch });
  }

  function setS(patch: Partial<SearchSettings>) {
    setSearch((cur) => ({ ...cur, ...patch }));
  }

  async function saveSearch() {
    setLoading("save-search");
    try {
      await api<PublicSearchSettings>("/api/settings/search", {
        method: "PUT",
        body: JSON.stringify(search),
      });
      onMessage("联网搜索设置已保存。", "info");
      setSearchKeySaved(true);
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "保存失败", "error");
    } finally {
      setLoading("");
    }
  }

  async function testSearch() {
    setLoading("test-search");
    try {
      const result = await api<{ ok: boolean; results: number; sample: string[] }>(
        "/api/settings/search/test",
        { method: "POST", body: JSON.stringify(search) },
      );
      onMessage(`搜索可用：返回 ${result.results} 条。示例：${result.sample.join(" / ") || "无"}`, "info");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "搜索测试失败", "error");
    } finally {
      setLoading("");
    }
  }

  async function initData() {
    setLoading("init-data");
    onMessage("正在下载并导入历史数据，可能需要 1-2 分钟，请稍候…", "info");
    try {
      const result = await api<WorldCupDataImportResult>("/api/ingest/run", { method: "POST" });
      onMessage(
        `数据初始化完成：世界杯 ${result.imported_records} 场，已为 ${result.teams_with_elo} 支球队建立 Elo。${result.message}`,
        "info",
      );
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "数据初始化失败", "error");
    } finally {
      setLoading("");
    }
  }

  async function save() {
    setLoading("save");
    try {
      await api<PublicLLMSettings>("/api/settings/llm", {
        method: "PUT",
        body: JSON.stringify(settings),
      });
      onMessage("LLM API 设置已保存到后端，并同步保留在本机浏览器。", "info");
      onSaved();
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "保存失败", "error");
    } finally {
      setLoading("");
    }
  }

  async function test() {
    setLoading("test");
    try {
      const result = await api<{ ok: boolean; message: string }>("/api/settings/llm/test", {
        method: "POST",
        body: JSON.stringify(settings),
      });
      onMessage(`LLM 连接正常：${result.message}`, "info");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "连接失败", "error");
    } finally {
      setLoading("");
    }
  }

  return (
    <div className="grid-2">
      <section className="card">
        <div className="page-heading compact">
          <div>
            <h2>LLM API 配置</h2>
            <p>这里修改的 API URL、模型名和开关会持久化保存；刷新页面后仍会保留。</p>
          </div>
        </div>
        <div className="form-stack">
          <label>
            API Base URL
            <input value={settings.base_url} onChange={(event) => set({ base_url: event.target.value })} />
          </label>
          <label>
            模型名
            <input
              value={settings.model}
              placeholder="例如 gpt-4.1-mini / 本地代理模型"
              onChange={(event) => set({ model: event.target.value })}
            />
          </label>
          <label>
            API Key
            <input
              type="password"
              value={settings.api_key}
              placeholder={apiKeySaved ? "后端已有 key；留空保存不会覆盖" : "请输入 API key"}
              onChange={(event) => set({ api_key: event.target.value })}
            />
            <small>
              {apiKeySaved
                ? "后端已保存 API key。若此处留空，保存时会继续沿用后端已保存的 key。"
                : "API key 会保存到后端 DuckDB；本机浏览器也会保留你当前填写的值，方便刷新后继续编辑。"}
            </small>
          </label>
          <div className="grid-2 tight">
            <label>
              Temperature
              <input
                type="number"
                min="0"
                max="2"
                step="0.1"
                value={settings.temperature}
                onChange={(event) => set({ temperature: Number(event.target.value) })}
              />
            </label>
            <label>
              超时秒数
              <input
                type="number"
                min="1"
                value={settings.timeout_seconds}
                onChange={(event) => set({ timeout_seconds: Number(event.target.value) })}
              />
            </label>
          </div>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={settings.enabled}
              onChange={(event) => set({ enabled: event.target.checked })}
            />
            启用 LLM 赔率清洗和多智能体解释报告
          </label>
          <div className="btn-row">
            <button className="btn btn-primary" onClick={save} disabled={loading === "save"}>
              <Save size={18} />
              保存设置
            </button>
            <button className="btn btn-ghost" onClick={test} disabled={loading === "test"}>
              <Activity size={18} />
              测试连接
            </button>
          </div>
        </div>
      </section>

      <section className="card">
        <div className="page-heading compact">
          <div>
            <h2><Globe size={18} /> 联网搜索配置</h2>
            <p>新闻舆情智能体用它获取球队伤停/阵容消息。与 LLM 解耦，任何供应商都能用。</p>
          </div>
        </div>
        <div className="form-stack">
          <label>
            搜索服务商
            <select
              value={search.provider}
              onChange={(event) => {
                const provider = event.target.value as SearchSettings["provider"];
                setS({ provider, base_url: SEARCH_PRESETS[provider] || search.base_url });
              }}
            >
              <option value="bocha">博查 Bocha（国内可达，推荐）</option>
              <option value="zhipu">智谱 Web Search</option>
              <option value="custom">自定义（Bocha 兼容）</option>
              <option value="none">不启用</option>
            </select>
          </label>
          <label>
            搜索 API Base URL
            <input value={search.base_url} onChange={(event) => setS({ base_url: event.target.value })} />
          </label>
          <label>
            搜索 API Key
            <input
              type="password"
              value={search.api_key}
              placeholder={searchKeySaved ? "后端已有 key；留空保存不会覆盖" : "请输入搜索 API key"}
              onChange={(event) => setS({ api_key: event.target.value })}
            />
          </label>
          <div className="grid-2 tight">
            <label>
              返回条数
              <input
                type="number"
                min="1"
                max="20"
                value={search.max_results}
                onChange={(event) => setS({ max_results: Number(event.target.value) })}
              />
            </label>
            <label>
              超时秒数
              <input
                type="number"
                min="1"
                value={search.timeout_seconds}
                onChange={(event) => setS({ timeout_seconds: Number(event.target.value) })}
              />
            </label>
          </div>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={search.enabled}
              onChange={(event) => setS({ enabled: event.target.checked })}
            />
            启用联网搜索（关闭时新闻智能体会明确提示数据源不可用）
          </label>
          <div className="btn-row">
            <button className="btn btn-primary" onClick={saveSearch} disabled={loading === "save-search"}>
              <Save size={18} /> 保存搜索设置
            </button>
            <button className="btn btn-ghost" onClick={testSearch} disabled={loading === "test-search"}>
              <Activity size={18} /> 测试搜索
            </button>
          </div>
        </div>
      </section>

      <section className="card">
        <div className="page-heading compact">
          <div>
            <h2><Database size={18} /> 数据初始化</h2>
            <p>首次使用请点击导入：下载世界杯历史 + 4.9 万场国际比赛，建立真实 Elo 和近期状态。</p>
          </div>
        </div>
        <button className="btn btn-primary" onClick={initData} disabled={loading === "init-data"}>
          <Database size={18} /> {loading === "init-data" ? "导入中…" : "下载并初始化数据"}
        </button>
        <p className="hint">未导入时，实力/近期状态智能体会缺少数据。导入约需 1-2 分钟，仅需一次。</p>
      </section>

      <section className="card help-card">
        <h2>使用流程</h2>
        <ol>
          <li>填入 LLM 的 API URL、模型名和 Key，测试连接后保存。</li>
          <li>可选：配置联网搜索（博查/智谱），让新闻智能体获取伤停消息。</li>
          <li>点击「下载并初始化数据」导入历史比赛，建立 Elo 与近期状态。</li>
          <li>回到首页或预测工作台开始使用。</li>
        </ol>
        <p>未启用 LLM 时，数值预测和赔率抓取仍可运行，赔率会使用规则清洗，解释报告会降级为模板说明。</p>
      </section>
    </div>
  );
}
