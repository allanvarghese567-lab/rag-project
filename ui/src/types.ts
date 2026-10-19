export type TabId = "api" | "llms" | "emb" | "jobs" | "conn";

export interface SessionConfig {
  apiBase: string;
  owner: string;
  repo: string;
  token: string;
  branch: string;
}

export interface StatusResponse {
  chunks: number;
  sources: string[];
  embedding_profile: Record<string, unknown>;
  llm_profile: Record<string, unknown>;
  store_dir: string;
}

export interface AskHit {
  content: string;
  source: string;
  score?: number;
}

export interface AskResponse {
  answer: string;
  hits: AskHit[];
  backend: string;
  k: number;
  llm_profile?: string | null;
}

export interface LlmProfile {
  id: string;
  provider: string;
  model: string;
  [key: string]: unknown;
}

export interface LlmsConfig {
  active: string;
  profiles: LlmProfile[];
}

export interface EmbeddingsConfig {
  active: string;
  profiles: Array<{ id: string; provider: string; model: string }>;
}

export interface TraceSpan {
  name: string;
  ms: number;
  data?: Record<string, unknown>;
}

export interface Trace {
  id?: string;
  name: string;
  total_ms: number;
  spans?: TraceSpan[];
}

export interface TracesResponse {
  traces: Trace[];
}

export interface JobStatus {
  ok?: boolean;
  status?: string;
  conclusion?: string | null;
  run_id?: number | null;
  run_number?: number | null;
  html_url?: string;
  message?: string;
}
