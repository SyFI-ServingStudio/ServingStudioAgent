import type {
  Conversation,
  ConversationListResponse,
  SandboxMode,
  Tokens,
  TurnEvent,
} from "./types";

export const DEFAULT_SANDBOX: SandboxMode = "workspace-write";

export async function listConversations(): Promise<ConversationListResponse> {
  const response = await fetch("/api/conversations");
  if (!response.ok) {
    throw new Error(`failed to list conversations: ${response.status}`);
  }
  return response.json();
}

export async function createConversation(
  sandbox: SandboxMode,
  autonomous: boolean,
): Promise<Conversation> {
  const response = await fetch("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sandbox, autonomous }),
  });
  if (!response.ok) {
    throw new Error(`failed to create conversation: ${response.status}`);
  }
  return response.json();
}

export async function getConversation(id: string): Promise<Conversation | null> {
  const response = await fetch(`/api/conversations/${id}`);
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`failed to load conversation: ${response.status}`);
  }
  return response.json();
}

export async function deleteConversation(id: string): Promise<void> {
  const response = await fetch(`/api/conversations/${id}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(`failed to delete conversation: ${response.status}`);
  }
}

export interface StreamHandlers {
  session?: (data: unknown) => void;
  progress?: (text: string) => void;
  /** One render-relevant turn event (intermediate_output/decision/usage/implementer/final). */
  event?: (event: TurnEvent) => void;
  done?: (text: string) => void;
}

export async function streamTurn(
  conversationId: string,
  text: string,
  sandbox: SandboxMode,
  autonomous: boolean,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`/api/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, sandbox_mode: sandbox, autonomous_mode: autonomous }),
    signal,
  });
  await consumeTurnStream(response, handlers);
}

export async function resumeTurn(
  conversationId: string,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<boolean> {
  const response = await fetch(`/api/conversations/${conversationId}/stream`, {
    signal,
  });
  if (response.status === 409) {
    return false;
  }
  await consumeTurnStream(response, handlers);
  return true;
}

export async function cancelTurn(conversationId: string): Promise<boolean> {
  const response = await fetch(`/api/conversations/${conversationId}/cancel`, {
    method: "POST",
  });
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
    handlers.done?.(`(request failed: ${response.status})`);
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
    case "progress":
      handlers.progress?.(String(data.text || ""));
      break;
    case "intermediate_output":
      handlers.event?.({
        kind: "intermediate_output",
        role: data.role ? String(data.role) : "",
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
        duration_ms: typeof data.duration_ms === "number" ? data.duration_ms : 0,
        tokens: parseTokens(data.tokens),
      });
      break;
    case "implementer":
      handlers.event?.({ kind: "implementer", text: String(data.text || "") });
      break;
    case "done":
      handlers.event?.({ kind: "final", text: String(data.text || "") });
      handlers.done?.(String(data.text || ""));
      break;
  }
}
