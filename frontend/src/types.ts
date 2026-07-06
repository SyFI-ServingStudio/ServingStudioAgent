export type SandboxMode = "read-only" | "workspace-write" | "danger-full-access";

export interface ConversationSummary {
  id: string;
  title: string;
  updated_at?: string;
}

export interface IntermediateOutput {
  role?: string;
  text: string;
}

export interface ChatMessage {
  role: "user" | "assistant" | string;
  content: string;
  intermediate_outputs?: IntermediateOutput[] | null;
}

export interface Conversation {
  id: string;
  title: string;
  sandbox?: SandboxMode | string;
  autonomous?: boolean;
  messages: ChatMessage[];
  codex_sessions?: Record<string, string>;
}

export interface ConversationListResponse {
  conversations: ConversationSummary[];
  sandbox_modes: SandboxMode[];
}

export interface StreamState {
  intermediateOutputs: IntermediateOutput[];
  progress: string;
  implementer: string;
  final: string;
  stopped: boolean;
}
