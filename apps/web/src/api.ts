const API_BASE_STORAGE_KEY = "world_cup_forecast_api_base";

function resolveApiBase(): string {
  const envBase = import.meta.env.VITE_API_BASE;
  if (envBase) return envBase.replace(/\/$/, "");

  try {
    const saved = localStorage.getItem(API_BASE_STORAGE_KEY);
    if (saved) return saved.replace(/\/$/, "");
  } catch {
    // Ignore storage failures and use host detection.
  }

  if (typeof window !== "undefined" && window.location.hostname) {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }

  return "http://127.0.0.1:8000";
}

export const API_BASE = resolveApiBase();

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

export type StreamHandlers = {
  onPrediction?: (data: any) => void;
  onReasoning?: (data: AgentReasoning) => void;
  onReport?: (data: { explanation: string; report_id: string }) => void;
  onDone?: (data: { report_id: string }) => void;
  onNodeStart?: (data: { node: string }) => void;
  onNodeEnd?: (data: { node: string }) => void;
  onError?: (error: Error) => void;
};

/**
 * Stream a multi-agent prediction via Server-Sent Events, surfacing each
 * agent's reasoning trace as it is produced.
 */
export async function streamPrediction(
  body: Record<string, unknown>,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/predict/match/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => "");
    throw new Error(text || response.statusText);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (event: string, data: string) => {
    let parsed: any;
    try {
      parsed = JSON.parse(data);
    } catch {
      return;
    }
    if (event === "prediction") handlers.onPrediction?.(parsed);
    else if (event === "reasoning") handlers.onReasoning?.(parsed as AgentReasoning);
    else if (event === "report") handlers.onReport?.(parsed);
    else if (event === "done") handlers.onDone?.(parsed);
    else if (event === "node_start") handlers.onNodeStart?.(parsed);
    else if (event === "node_end") handlers.onNodeEnd?.(parsed);
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      let event = "message";
      let data = "";
      for (const line of chunk.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) dispatch(event, data);
    }
  }
}

export type AskResponse = {
  question: string;
  home_team: string | null;
  away_team: string | null;
  answer: string;
  matched: boolean;
};

export type TodayCard = {
  match_id: string;
  home_team: string;
  away_team: string;
  kickoff_time?: string | null;
  status?: string | null;
  probabilities: { home_win: number; draw: number; away_win: number };
  most_likely_score: string;
  recommendation: BetSignal | null;
  has_value: boolean;
};

export type TodayOverview = {
  count: number;
  cards: TodayCard[];
};

export type Health = {
  status: string;
  db_path: string;
  odds_records: number;
  match_records: number;
  live_matches: number;
  standing_records: number;
  llm_enabled: boolean;
  model_version: string;
  scheduler_running: boolean;
};

export type LLMSettings = {
  base_url: string;
  api_key: string;
  model: string;
  temperature: number;
  timeout_seconds: number;
  enabled: boolean;
};

export type PublicLLMSettings = Omit<LLMSettings, "api_key"> & {
  api_key_masked: string;
  api_key_saved: boolean;
};

export type SearchSettings = {
  provider: "none" | "bocha" | "zhipu" | "custom";
  base_url: string;
  api_key: string;
  timeout_seconds: number;
  max_results: number;
  enabled: boolean;
};

export type PublicSearchSettings = Omit<SearchSettings, "api_key"> & {
  api_key_masked: string;
  api_key_saved: boolean;
};

export type OddsRecord = {
  match_id: string;
  kickoff_time?: string;
  home_team: string;
  away_team: string;
  play_type: string;
  handicap?: string;
  win_odds?: number;
  draw_odds?: number;
  lose_odds?: number;
  source: string;
  source_url: string;
  scraped_at: string;
  raw?: {
    cleaning?: { method?: string };
    llm_cleaning?: { reason?: string; home_before?: string; away_before?: string };
  };
};

export type AgentFinding = {
  agent: string;
  confidence: number;
  signal: "positive" | "neutral" | "negative";
  rationale: string;
  sources: string[];
  metrics?: Record<string, number>;
};

export type ReasoningStep = {
  kind: "observation" | "analysis" | "conclusion";
  content: string;
};

export type AgentReasoning = {
  agent: string;
  confidence: number;
  signal: "positive" | "neutral" | "negative";
  steps: ReasoningStep[];
  rationale: string;
  sources: string[];
  metrics?: Record<string, number>;
  powered_by: "llm" | "deterministic";
};

export type BetSignal = {
  outcome: string;
  model_probability: number;
  market_probability?: number;
  odds?: number;
  edge?: number;
  kelly_fraction: number;
  stake: number;
  rationale: string;
};

export type Prediction = {
  probabilities: { home_win: number; draw: number; away_win: number };
  expected_score: [number, number];
  most_likely_score: string;
  odds?: OddsRecord;
  bet_signals: BetSignal[];
  agent_findings: AgentFinding[];
  explanation: string;
  shap_values?: Record<string, number>;
};

export type TournamentTeamProbability = {
  team: string;
  group_advance: number;
  quarter_final: number;
  semi_final: number;
  final: number;
  champion: number;
};

export type TournamentBracket = {
  teams: string[];
  simulations: number;
  results: TournamentTeamProbability[];
  fixtures_used: number;
  current_state_used: boolean;
  group_results?: Record<string, { teams: string[]; fixtures: number; current_state_used: boolean }>;
};

export type BacktestMetrics = {
  tournament_year: number;
  brier: number;
  log_loss: number;
  roi: number;
  max_drawdown: number;
  record_count: number;
  baseline_brier: number;
  baseline_log_loss: number;
};

export type BacktestRunResult = {
  id: string;
  run_at: string;
  metrics: BacktestMetrics[];
  params: BacktestRunRequest;
  explanation?: string;
};

export type BacktestRunRequest = {
  years: number[];
  edge_threshold: number;
  kelly_fraction: number;
  max_stake_fraction: number;
  initial_bankroll: number;
  home_odds: number;
  draw_odds: number;
  away_odds: number;
};

export type WorldCupDataImportResult = {
  status: string;
  source_url: string;
  raw_path: string;
  output_path: string;
  rows: number;
  years: number[];
  imported_records: number;
  teams_with_elo: number;
  message: string;
};

export type LiveMatch = {
  match_id: string;
  name: string;
  short_name: string;
  date: string;
  stage: string;
  status_state: string;
  status_name: string;
  completed: boolean;
  display_clock: string;
  venue: string;
  home_team: string;
  home_abbr: string;
  home_score: number | null;
  away_team: string;
  away_abbr: string;
  away_score: number | null;
  home_odds: number | null;
  away_odds: number | null;
  fetched_at: string;
};

export type ESPNTeam = {
  team_id: string;
  name: string;
  short_name: string;
  abbreviation: string;
  location: string;
  color: string;
  logo_url: string;
};

export type GroupStandingRecord = {
  group_name: string;
  rank: number;
  team: string;
  played: number;
  wins: number;
  draws: number;
  losses: number;
  goals_for: number;
  goals_against: number;
  goal_difference: number;
  points: number;
  source: string;
  source_url: string;
  scraped_at: string;
};

export type TournamentFixture = {
  match_id: string;
  home_team: string;
  away_team: string;
  group_name?: string | null;
  round_name: string;
  kickoff_time?: string | null;
  completed: boolean;
  home_score?: number | null;
  away_score?: number | null;
  source: string;
};

export type TournamentScheduleImport = {
  teams: string[];
  fixtures: TournamentFixture[];
  standings: GroupStandingRecord[];
  imported_at: string;
};
