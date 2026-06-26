import React, { useState } from "react";
import { Activity, Save } from "lucide-react";
import { api, type LLMSettings, type PublicLLMSettings } from "../api";

type Props = {
  settings: LLMSettings;
  apiKeySaved: boolean;
  onChange: (settings: LLMSettings) => void;
  onMessage: (message: string, type?: "info" | "error") => void;
  onSaved: () => void;
};

export function Settings({ settings, apiKeySaved, onChange, onMessage, onSaved }: Props) {
  const [loading, setLoading] = useState("");

  function set(patch: Partial<LLMSettings>) {
    onChange({ ...settings, ...patch });
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

      <section className="card help-card">
        <h2>使用流程</h2>
        <ol>
          <li>填入 OpenAI-compatible API URL、模型名和 API Key。</li>
          <li>点击“测试连接”确认当前配置可用。</li>
          <li>点击“保存设置”，配置会写入后端 DuckDB，并在本机浏览器保留一份。</li>
          <li>回到预测工作台，启用 LLM 后即可生成多智能体中文解释报告。</li>
        </ol>
        <p>未启用 LLM 时，数值预测和赔率抓取仍可运行，赔率会使用规则清洗，解释报告会降级为模板说明。</p>
      </section>
    </div>
  );
}
