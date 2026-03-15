export interface GenerationStats {
  prefill_tokens: number;
  decode_tokens: number;
  prefill_ms: number;
  decode_ms: number;
  tokens_per_sec: number;
}

export interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  images?: string[];
  stats?: GenerationStats;
}

export interface ConversationSettings {
  temperature: number;
  top_k: number;
  top_p: number;
  max_tokens: number;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  system_prompt: string;
  messages: Message[];
  settings: ConversationSettings;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ModelInfo {
  name: string;
  path: string;
  backend: string;
  num_layers: number;
  hidden_size: number;
  vocab_size: number;
}

export interface AppSettings {
  model_path: string | null;
  backend: string;
  default_system_prompt: string;
  default_settings: ConversationSettings;
  nothink: boolean;
}

export type LoadingEvent =
  | { type: "progress"; step: string; percent: number }
  | { type: "done" }
  | { type: "error"; message: string };

export type TokenEvent =
  | { type: "delta"; text: string }
  | { type: "done"; stats: GenerationStats }
  | { type: "cancelled" }
  | { type: "error"; message: string };

export interface PendingImage {
  id: string;
  thumbnailUrl: string;
  path: string;
  status: "encoding" | "ready" | "error";
  progress: number;
  label: string;
  numTokens?: number;
}

export type VisionEvent =
  | { type: "progress"; image_id: string; percent: number; label: string }
  | { type: "done"; image_id: string; num_tokens: number }
  | { type: "error"; image_id: string; message: string };
