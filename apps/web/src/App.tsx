import React, { useEffect, useState } from "react";
import { BarChart3, Brain, Gauge, Home as HomeIcon, Settings as SettingsIcon, TestTube2, Trophy } from "lucide-react";
import { api, type Health, type LLMSettings, type LiveMatch, type OddsRecord, type PublicLLMSettings } from "./api";
import { Backtest } from "./components/Backtest";
import { Header } from "./components/Header";
import { Home } from "./components/Home";
import { OddsLog } from "./components/OddsLog";
import { Settings } from "./components/Settings";
import { Tournament } from "./components/Tournament";
import { Workbench } from "./components/Workbench";

const SETTINGS_STORAGE_KEY = "world_cup_forecast_llm_settings";

const DEFAULT_LLM_SETTINGS: LLMSettings = {
  base_url: "https://api.openai.com/v1",
  api_key: "",
  model: "gpt-4.1-mini",
  temperature: 0.2,
  timeout_seconds: 30,
  enabled: false,
};

const TABS = [
  { id: "home", label: "首页", icon: HomeIcon },
  { id: "predict", label: "预测工作台", icon: Brain },
  { id: "odds", label: "500赔率", icon: Gauge },
  { id: "tourney", label: "锦标赛模拟", icon: Trophy },
  { id: "backtest", label: "参数回测", icon: TestTube2 },
  { id: "settings", label: "系统设置", icon: SettingsIcon },
] as const;

type TabId = (typeof TABS)[number]["id"];

function loadLocalSettings(): LLMSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return DEFAULT_LLM_SETTINGS;
    return { ...DEFAULT_LLM_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_LLM_SETTINGS;
  }
}

function saveLocalSettings(settings: LLMSettings) {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Browser storage can be unavailable in private or restricted modes.
  }
}

export function App() {
  const [tab, setTab] = useState<TabId>("home");
  const [health, setHealth] = useState<Health | null>(null);
  const [odds, setOdds] = useState<OddsRecord[]>([]);
  const [liveMatches, setLiveMatches] = useState<LiveMatch[]>([]);
  const [message, setMessage] = useState("");
  const [msgType, setMsgType] = useState<"info" | "error">("info");
  const [settings, setSettings] = useState<LLMSettings>(() => loadLocalSettings());
  const [apiKeySaved, setApiKeySaved] = useState(false);

  function notify(msg: string, type: "info" | "error" = "info") {
    setMessage(msg);
    setMsgType(type);
  }

  function updateSettings(next: LLMSettings) {
    setSettings(next);
    saveLocalSettings(next);
  }

  async function refresh() {
    try {
      const [h, o, llm, lm] = await Promise.all([
        api<Health>("/api/health"),
        api<OddsRecord[]>("/api/odds/latest?limit=200"),
        api<PublicLLMSettings>("/api/settings/llm"),
        api<LiveMatch[]>("/api/espn/matches"),
      ]);
      setHealth(h);
      setOdds(o);
      setLiveMatches(lm);
      setApiKeySaved(llm.api_key_saved);
      setSettings((prev) => {
        const merged = {
          ...prev,
          base_url: llm.base_url,
          model: llm.model,
          temperature: llm.temperature,
          timeout_seconds: llm.timeout_seconds,
          enabled: llm.enabled,
        };
        saveLocalSettings(merged);
        return merged;
      });
    } catch (error) {
      notify(error instanceof Error ? error.message : "无法连接后端服务，已保留本机缓存的 API 设置。", "error");
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="app-shell">
      <Header health={health} onRefresh={refresh} />

      <nav className="tab-nav" aria-label="主功能导航">
        {TABS.map((tabItem) => {
          const Icon = tabItem.icon;
          return (
            <button
              key={tabItem.id}
              className={`tab-btn${tab === tabItem.id ? " active" : ""}`}
              onClick={() => setTab(tabItem.id)}
            >
              <Icon size={18} />
              {tabItem.label}
            </button>
          );
        })}
      </nav>

      <main className="page">
        {message && (
          <button className={`notice${msgType === "error" ? " error" : ""}`} onClick={() => setMessage("")}>
            <BarChart3 size={18} />
            <span>{message}</span>
            <strong>关闭</strong>
          </button>
        )}

        {tab === "home" && (
          <Home onMessage={(m, type = "info") => notify(m, type)} />
        )}
        {tab === "predict" && (
          <Workbench
            liveMatches={liveMatches}
            onRefreshMatches={refresh}
            onMessage={(m, type = "error") => notify(m, type)}
          />
        )}
        {tab === "odds" && (
          <OddsLog odds={odds} onRefresh={setOdds} onMessage={(m, type = "info") => notify(m, type)} />
        )}
        {tab === "tourney" && <Tournament />}
        {tab === "backtest" && <Backtest />}
        {tab === "settings" && (
          <Settings
            settings={settings}
            apiKeySaved={apiKeySaved}
            onChange={updateSettings}
            onMessage={(m, type = "info") => notify(m, type)}
            onSaved={refresh}
          />
        )}
      </main>
    </div>
  );
}
