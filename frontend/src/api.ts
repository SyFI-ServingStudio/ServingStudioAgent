import type {
  Conversation,
  ConversationListResponse,
  IntermediateOutput,
  SandboxMode,
} from "./types";

export const DEFAULT_SANDBOX: SandboxMode = "workspace-write";

export async function listConversations(): Promise<ConversationListResponse> {
  const response = await fetch("/api/conversations");
  if (!response.ok) {
    throw new Error(`failed to list conversations: ${response.status}`);
  }
  return response.json();
}

export async function createConversation(sandbox: SandboxMode): Promise<Conversation> {
  const response = await fetch("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sandbox }),
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
  intermediateOutput?: (output: IntermediateOutput) => void;
  implementer?: (text: string) => void;
  done?: (text: string) => void;
}

export async function streamTurn(
  conversationId: string,
  text: string,
  sandbox: SandboxMode,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`/api/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, sandbox_mode: sandbox }),
    signal,
  });
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

  if (event === "session") {
    handlers.session?.(data);
  } else if (event === "progress") {
    handlers.progress?.(String(data.text || ""));
  } else if (event === "intermediate_output") {
    handlers.intermediateOutput?.({
      role: data.role ? String(data.role) : "",
      text: data.text ? String(data.text) : "",
    });
  } else if (event === "implementer") {
    handlers.implementer?.(String(data.text || ""));
  } else if (event === "done") {
    handlers.done?.(String(data.text || ""));
  }
}
