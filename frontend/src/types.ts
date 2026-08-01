export type SandboxMode = "read-only" | "workspace-write" | "danger-full-access";

export type Role = "orchestrator" | "implementer";
export type CommentaryLevel = "progress" | "milestone";
/** One role's Codex choice: which model runs it, and at what reasoning effort. */
export interface CodexRoleRuntime {
  model: string;
  effort: string;
}

export interface CodexRuntimeSelection {
  orchestrator: CodexRoleRuntime;
  implementer: CodexRoleRuntime;
}

export interface CodexModelOption {
  id: string;
  label: string;
  /** Session-compatibility boundary: a started conversation may only move within it. */
  family: string;
  familyLabel: string;
  efforts: string[];
  defaultEffort: string;
  available: boolean;
}

export interface CodexRuntimeCatalog {
  models: CodexModelOption[];
  defaults: CodexRuntimeSelection;
}

export interface ConversationSummary {
  id: string;
  title: string;
  updated_at?: number | string;
  workspace_id?: string;
}

export interface IntermediateOutput {
  role?: string;
  level?: CommentaryLevel;
  text: string;
}

export interface Tokens {
  read: number;
  prefill: number;
  output: number;
}

export type TerminalOutcome = "final_answer" | "request_user_input";

/**
 * Render-relevant turn events. Mirrors the backend `activity` list persisted on
 * an assistant message and the SSE events streamed during a live turn, so the
 * same `reduceTurn` reducer drives both playback and reload.
 */
export type TurnEvent =
  | { kind: "intermediate_output"; role: string; model?: string; effort?: string; level?: CommentaryLevel; text: string }
  | { kind: "decision"; action: string; task: string }
  | { kind: "implementer"; text: string }
  | { kind: "usage"; role: string; model?: string; effort?: string; duration_ms: number; tokens: Tokens }
  | { kind: "final"; text: string; outcome?: TerminalOutcome };

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
  codex_runtime?: CodexRuntimeSelection;
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
  runtime?: CodexRoleRuntime;
  round: number;
  notes: RoleNote[];
  durationMs: number | null;
  tokens: Tokens | null;
  done: boolean;
}

export interface RoleNote {
  level: CommentaryLevel;
  text: string;
}

export interface Handoff {
  type: "handoff";
  variant: "delegated_task" | "conclusion";
  text: string;
}

export interface TerminalResponse {
  type: "response";
  text: string;
  outcome: TerminalOutcome;
}

export type TurnCard = RolePhase | Handoff | TerminalResponse;
