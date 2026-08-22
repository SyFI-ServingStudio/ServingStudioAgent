export type SandboxMode = "read-only" | "workspace-write" | "danger-full-access";

export type Role = "orchestrator" | "implementer" | "assistant";
/**
 * How many Codex backends drive a turn. `orchestrated` is the two-role
 * orchestrator/implementer loop; `single` is one `assistant` that both plans and
 * implements. Conversation-level and pinned after the first message, because the
 * Codex sessions a turn builds are per role.
 */
export type AgentMode = "orchestrated" | "single";
export type CommentaryLevel = "progress" | "milestone";
export type CodexServiceTier = "default" | "fast";

/** One role's Codex choice: model, reasoning effort, and per-call speed tier. */
export interface CodexRoleRuntime {
  model: string;
  effort: string;
  serviceTier: CodexServiceTier;
}

/** Every role is carried, whichever mode is active; the unused one stays parked. */
export interface CodexRuntimeSelection {
  orchestrator: CodexRoleRuntime;
  implementer: CodexRoleRuntime;
  assistant: CodexRoleRuntime;
}

export interface CodexModelOption {
  id: string;
  label: string;
  /** Session-compatibility boundary: a started conversation may only move within it. */
  family: string;
  familyLabel: string;
  efforts: string[];
  defaultEffort: string;
  serviceTiers: CodexServiceTier[];
  defaultServiceTier: CodexServiceTier;
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
  | { kind: "error"; text: string; code?: string }
  | { kind: "final"; text: string; outcome?: TerminalOutcome };

/** Why a turn ended without an answer. Set instead of an outcome, never beside one. */
export interface TurnFailure {
  code: string;
  message?: string;
}

export interface ChatMessage {
  role: "user" | "assistant" | string;
  content: string;
  intermediate_outputs?: IntermediateOutput[] | null;
  activity?: TurnEvent[] | null;
  failure?: TurnFailure | null;
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
  agent_mode?: AgentMode;
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
  /** Whichever role emitted it — decides the "Input needed" card's tint. */
  role: Role;
}

/** A turn that ended without an answer. Replaces the response card, never joins it. */
export interface FailureNotice {
  type: "failure";
  text: string;
  code?: string;
}

export type TurnCard = RolePhase | Handoff | TerminalResponse | FailureNotice;
