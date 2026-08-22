import type {
  AgentMode,
  CodexRuntimeCatalog,
  CodexRuntimeSelection,
  Conversation,
  ConversationListResponse,
  SandboxMode,
  TerminalOutcome,
  Tokens,
  TurnEvent,
} from "./types";
import { conversationLocation, conversationLocator } from "./conversationLocator";

export const DEFAULT_SANDBOX: SandboxMode = "workspace-write";
export const MESSAGE_PAGE_SIZE = 20;
const MAIN_WORKSPACE_ID = "w_main";

export async function listCodexBackends(): Promise<CodexRuntimeCatalog> {
  const response = await fetch("/api/codex-backends");
  if (!response.ok) {
    throw new Error(`failed to list Codex backends: ${response.status}`);
  }
  return response.json();
}

function locatedConversation(
  conversation: Conversation,
  workspaceId: string,
): Conversation {
  return {
    ...conversation,
    id: conversationLocator(workspaceId, conversation.id),
  };
}

export async function listConversations(): Promise<ConversationListResponse> {
  const response = await fetch("/api/conversations");
  if (!response.ok) {
    throw new Error(`failed to list conversations: ${response.status}`);
  }
  const payload = (await response.json()) as ConversationListResponse;
  return {
    ...payload,
    conversations: (payload.conversations || []).map((conversation) => ({
      ...conversation,
      id: conversationLocator(
        conversation.workspace_id || MAIN_WORKSPACE_ID,
        conversation.id,
      ),
    })),
  };
}

export async function createConversation(
  sandbox: SandboxMode,
  autonomous: boolean,
  agentMode: AgentMode,
  codexRuntime: CodexRuntimeSelection,
): Promise<Conversation> {
  const response = await fetch(`/api/workspaces/${MAIN_WORKSPACE_ID}/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sandbox,
      autonomous,
      agent_mode: agentMode,
      codex_runtime: codexRuntime,
    }),
  });
  if (!response.ok) {
    throw new Error(`failed to create conversation: ${response.status}`);
  }
  return locatedConversation(await response.json(), MAIN_WORKSPACE_ID);
}

export async function getConversation(
  id: string,
  before?: number,
): Promise<Conversation | null> {
  const location = conversationLocation(id);
  const query = new URLSearchParams({ limit: String(MESSAGE_PAGE_SIZE) });
  if (before !== undefined) {
    query.set("before", String(before));
  }
  const response = await fetch(
    `/api/workspaces/${encodeURIComponent(location.workspaceId)}/conversations/${encodeURIComponent(location.conversationId)}?${query}`,
  );
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`failed to load conversation: ${response.status}`);
  }
  return locatedConversation(await response.json(), location.workspaceId);
}

export async function updateConversationRuntime(
  id: string,
  codexRuntime: CodexRuntimeSelection,
): Promise<Conversation> {
  const location = conversationLocation(id);
  const response = await fetch(
    `/api/workspaces/${encodeURIComponent(location.workspaceId)}/conversations/${encodeURIComponent(location.conversationId)}/runtime`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ codex_runtime: codexRuntime }),
    },
  );
  if (!response.ok) throw new Error(`failed to update conversation runtime: ${response.status}`);
  return locatedConversation(await response.json(), location.workspaceId);
}

export async function deleteConversation(id: string): Promise<void> {
  const location = conversationLocation(id);
  const response = await fetch(
    `/api/workspaces/${encodeURIComponent(location.workspaceId)}/conversations/${encodeURIComponent(location.conversationId)}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    throw new Error(`failed to delete conversation: ${response.status}`);
  }
}

export interface StreamHandlers {
  session?: (data: unknown) => void;
  toolCall?: (text: string) => void;
  /** One render-relevant turn event (intermediate_output/decision/usage/implementer/final). */
  event?: (event: TurnEvent) => void;
  done?: (text: string, outcome: TerminalOutcome) => void;
}

export async function streamTurn(
  conversationId: string,
  text: string,
  sandbox: SandboxMode,
  autonomous: boolean,
  agentMode: AgentMode,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const location = conversationLocation(conversationId);
  const response = await fetch(
    `/api/workspaces/${encodeURIComponent(location.workspaceId)}/conversations/${encodeURIComponent(location.conversationId)}/messages`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        sandbox_mode: sandbox,
        autonomous_mode: autonomous,
        // Only honored on the first turn; the backend pins it afterwards.
        agent_mode: agentMode,
      }),
      signal,
    },
  );
  await consumeTurnStream(response, handlers);
}

export async function resumeTurn(
  conversationId: string,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<boolean> {
  const location = conversationLocation(conversationId);
  const response = await fetch(
    `/api/workspaces/${encodeURIComponent(location.workspaceId)}/conversations/${encodeURIComponent(location.conversationId)}/stream`,
    { signal },
  );
  // 204 is the normal idle response; 409 is accepted for compatibility with
  // an older backend during a rolling restart.
  if (response.status === 204 || response.status === 409) {
    return false;
  }
  await consumeTurnStream(response, handlers);
  return true;
}

export async function cancelTurn(conversationId: string): Promise<boolean> {
  const location = conversationLocation(conversationId);
  const response = await fetch(
    `/api/workspaces/${encodeURIComponent(location.workspaceId)}/conversations/${encodeURIComponent(location.conversationId)}/cancel`,
    { method: "POST" },
  );
  if (!response.ok) {
    throw new Error(`failed to stop turn: ${response.status}`);
  }
  const result = (await response.json()) as { cancelled?: boolean };
  return Boolean(result.cancelled);
}

async function consumeTurnStream(
  response: Response,
  handlers: StreamHandlers,
): Promise<void> {
  if (!response.ok || !response.body) {
    // Our own backend failed, so no `done` frame is coming. Emit the same
    // failure event the stream would have, or the turn renders as an answer.
    const text = `The request to the VibeSim backend failed (HTTP ${response.status}).`;
    handlers.event?.({ kind: "error", text, code: "backend_unreachable" });
    handlers.done?.(text, "final_answer");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      handleSseChunk(chunk, handlers);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function parseTokens(raw: unknown): Tokens {
  const source = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const num = (value: unknown): number => (typeof value === "number" ? value : 0);
  return { read: num(source.read), prefill: num(source.prefill), output: num(source.output) };
}

function handleSseChunk(chunk: string, handlers: StreamHandlers): void {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of chunk.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).replace(/^ /, ""));
    }
  }

  const rawData = dataLines.join("\n");
  let data: Record<string, unknown> = {};
  try {
    data = rawData ? JSON.parse(rawData) : {};
  } catch {
    data = {};
  }

  switch (event) {
    case "session":
      handlers.session?.(data);
      break;
    case "tool_call":
      handlers.toolCall?.(String(data.text || ""));
      break;
    case "intermediate_output":
      handlers.event?.({
        kind: "intermediate_output",
        role: data.role ? String(data.role) : "",
        model: data.model ? String(data.model) : undefined,
        effort: data.effort ? String(data.effort) : undefined,
        level: data.level === "milestone" ? "milestone" : "progress",
        text: data.text ? String(data.text) : "",
      });
      break;
    case "decision":
      handlers.event?.({
        kind: "decision",
        action: data.action ? String(data.action) : "",
        task: data.task ? String(data.task) : "",
      });
      break;
    case "usage":
      handlers.event?.({
        kind: "usage",
        role: data.role ? String(data.role) : "",
        model: data.model ? String(data.model) : undefined,
        effort: data.effort ? String(data.effort) : undefined,
        duration_ms: typeof data.duration_ms === "number" ? data.duration_ms : 0,
        tokens: parseTokens(data.tokens),
      });
      break;
    case "implementer":
      handlers.event?.({ kind: "implementer", text: String(data.text || "") });
      break;
    case "error":
      // The transient line only; the terminal failure card arrives on `done`
      // milliseconds later, and two red cards for one failure read as two.
      handlers.toolCall?.(String(data.text || ""));
      break;
    case "done":
      {
        const text = String(data.text || "");
        const failure = (data.failure ?? null) as { code?: string } | null;
        if (failure) {
          handlers.event?.({
            kind: "error",
            text,
            code: failure.code ? String(failure.code) : undefined,
          });
          handlers.done?.(text, "final_answer");
          break;
        }
        const outcome: TerminalOutcome =
          data.outcome === "request_user_input" ? "request_user_input" : "final_answer";
        handlers.event?.({ kind: "final", text, outcome });
        handlers.done?.(text, outcome);
      }
      break;
  }
}
