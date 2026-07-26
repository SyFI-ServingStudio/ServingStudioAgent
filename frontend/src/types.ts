export type SandboxMode = "read-only" | "workspace-write" | "danger-full-access";

export type Role = "orchestrator" | "implementer";

export interface ConversationSummary {
  id: string;
  title: string;
  updated_at?: string;
}

export interface IntermediateOutput {
  role?: string;
  text: string;
}

export interface Tokens {
  read: number;
  prefill: number;
  output: number;
}

/**
 * Render-relevant turn events. Mirrors the backend `activity` list persisted on
 * an assistant message and the SSE events streamed during a live turn, so the
 * same `reduceTurn` reducer drives both playback and reload.
 */
export type TurnEvent =
  | { kind: "intermediate_output"; role: string; text: string }
  | { kind: "decision"; action: string; task: string }
  | { kind: "implementer"; text: string }
  | { kind: "usage"; role: string; duration_ms: number; tokens: Tokens }
  | { kind: "final"; text: string };

export interface ChatMessage {
  role: "user" | "assistant" | string;
  content: string;
  intermediate_outputs?: IntermediateOutput[] | null;
  activity?: TurnEvent[] | null;
}

export interface MessagePage {
  start_index: number;
  end_index: number;
  total_messages: number;
  has_more: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  sandbox?: SandboxMode | string;
  autonomous?: boolean;
  messages: ChatMessage[];
  message_page?: MessagePage;
  codex_sessions?: Record<string, string>;
}

export interface ConversationListResponse {
  conversations: ConversationSummary[];
  sandbox_modes: SandboxMode[];
}

/** A single card in the rendered role timeline (output of `reduceTurn`). */
export interface RolePhase {
  type: "role";
  role: Role;
  round: number;
  notes: string[];
  durationMs: number | null;
  tokens: Tokens | null;
  done: boolean;
}

export interface Handoff {
  type: "handoff";
  variant: "delegated_task" | "conclusion";
  text: string;
}

export interface Answer {
  type: "answer";
  text: string;
}

export type TurnCard = RolePhase | Handoff | Answer;
