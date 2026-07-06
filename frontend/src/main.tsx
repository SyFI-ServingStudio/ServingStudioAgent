import React, { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

import {
  DEFAULT_SANDBOX,
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  streamTurn,
} from "./api";
import { markdownHtml, normalizeBackendText } from "./markdown";
import type {
  ChatMessage,
  Conversation,
  ConversationSummary,
  IntermediateOutput,
  SandboxMode,
  StreamState,
} from "./types";

import "./styles.css";

const EMPTY_STREAM: StreamState = {
  intermediateOutputs: [],
  progress: "",
  implementer: "",
  final: "",
  stopped: false,
};

const SUGGESTIONS = [
  {
    label: "List the L1 profilers",
    fill: "List the available MLSim L1 profilers.",
  },
  {
    label: "Count missing single_gemm rows",
    fill: "Count missing rows for single_gemm, backend torch.",
  },
  {
    label: "How do I run a simulation?",
    fill: "What presets are available to run a simulation, and how do I dry-run one?",
  },
];

function App() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [title, setTitle] = useState("MLSim Assistant");
  const [sandboxModes, setSandboxModes] = useState<SandboxMode[]>([
    "read-only",
    "workspace-write",
    "danger-full-access",
  ]);
  const [sandbox, setSandbox] = useState<SandboxMode>(
    (localStorage.getItem("mlsim_sandbox") as SandboxMode | null) || DEFAULT_SANDBOX,
  );
  const [autonomous, setAutonomous] = useState(
    localStorage.getItem("mlsim_autonomous") === "1",
  );
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [stream, setStream] = useState<StreamState | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const messagesRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    refreshSidebar().then((items) => {
      if (items.length) {
        void selectConversation(items[0].id);
      }
    });
  }, []);

  useEffect(() => {
    localStorage.setItem("mlsim_sandbox", sandbox);
  }, [sandbox]);

  useEffect(() => {
    localStorage.setItem("mlsim_autonomous", autonomous ? "1" : "0");
  }, [autonomous]);

  useEffect(() => {
    messagesRef.current?.scrollTo({
      top: messagesRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, stream]);

  async function refreshSidebar(): Promise<ConversationSummary[]> {
    const data = await listConversations();
    const items = data.conversations || [];
    setConversations(items);
    if (data.sandbox_modes?.length) {
      setSandboxModes(data.sandbox_modes);
    }
    return items;
  }

  async function selectConversation(id: string): Promise<void> {
    if (streaming) {
      return;
    }
    const conversation = await getConversation(id);
    if (!conversation) {
      await refreshSidebar();
      return;
    }
    loadConversation(conversation);
  }

  function loadConversation(conversation: Conversation): void {
    setCurrentId(conversation.id);
    setMessages(conversation.messages || []);
    if (conversation.sandbox) {
      setSandbox(conversation.sandbox as SandboxMode);
    }
    setAutonomous(Boolean(conversation.autonomous));
    setTitle(
      conversation.title && conversation.title !== "New chat"
        ? conversation.title
        : "MLSim Assistant",
    );
  }

  async function newConversation(): Promise<void> {
    if (streaming) {
      return;
    }
    const conversation = await createConversation(sandbox, autonomous);
    setCurrentId(conversation.id);
    setMessages([]);
    setTitle("MLSim Assistant");
    await refreshSidebar();
    inputRef.current?.focus();
  }

  async function removeConversation(id: string): Promise<void> {
    if (streaming) {
      return;
    }
    await deleteConversation(id);
    if (currentId === id) {
      setCurrentId(null);
      setMessages([]);
      setTitle("MLSim Assistant");
    }
    await refreshSidebar();
  }

  async function submitMessage(event?: FormEvent): Promise<void> {
    event?.preventDefault();
    const text = input.trim();
    if (streaming || !text) {
      return;
    }

    let conversationId = currentId;
    if (!conversationId) {
      const conversation = await createConversation(sandbox, autonomous);
      conversationId = conversation.id;
      setCurrentId(conversation.id);
      await refreshSidebar();
    }

    setInput("");
    setMessages((current) => [...current, { role: "user", content: text }]);
    setStream(EMPTY_STREAM);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;
    let localOnlyAssistant: ChatMessage | null = null;

    try {
      await streamTurn(
        conversationId,
        text,
        sandbox,
        autonomous,
        {
          progress: (line) => {
            if (!line) {
              return;
            }
            setStream((current) => ({ ...(current || EMPTY_STREAM), progress: line }));
          },
          intermediateOutput: (output) => {
            setStream((current) => ({
              ...(current || EMPTY_STREAM),
              intermediateOutputs: [...(current?.intermediateOutputs || []), output],
            }));
          },
          implementer: (text) => {
            setStream((current) => ({
              ...(current || EMPTY_STREAM),
              implementer: text || "(empty implementer summary)",
            }));
          },
          done: (answer) => {
            setStream((current) => ({
              ...(current || EMPTY_STREAM),
              final: answer,
            }));
          },
        },
        controller.signal,
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        localOnlyAssistant = { role: "assistant", content: "Stopped." };
        setStream((current) => ({ ...(current || EMPTY_STREAM), stopped: true }));
      } else {
        localOnlyAssistant = {
          role: "assistant",
          content: `(error talking to backend: ${String(error)})`,
        };
        setStream((current) => ({
          ...(current || EMPTY_STREAM),
          final: localOnlyAssistant?.content || "",
        }));
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
      if (localOnlyAssistant) {
        const assistantMessage = localOnlyAssistant;
        setMessages((current) => [...current, assistantMessage]);
      } else {
        const conversation = await getConversation(conversationId);
        if (conversation) {
          loadConversation(conversation);
        }
      }
      setStream(null);
      await refreshSidebar();
      inputRef.current?.focus();
    }
  }

  function stopTurn(): void {
    if (streaming && abortRef.current) {
      abortRef.current.abort();
    }
  }

  function handleTextareaKey(event: KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitMessage();
    }
  }

  function updateInput(value: string): void {
    setInput(value);
    requestAnimationFrame(() => {
      const textarea = inputRef.current;
      if (!textarea) {
        return;
      }
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    });
  }

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentId={currentId}
        onNewConversation={newConversation}
        onSelect={selectConversation}
        onDelete={removeConversation}
      />
      <main className="chat">
        <header className="chat-header">
          <div className="chat-title">{title}</div>
          <div className="chat-sub">
            drives <code>codex exec</code> inside a copied <code>main/</code> workspace
          </div>
        </header>

        <section ref={messagesRef} className="messages">
          {messages.length === 0 && !stream ? (
            <Welcome onSuggestion={(fill) => updateInput(fill)} />
          ) : (
            <>
              {messages.map((message, index) => (
                <MessageRow
                  key={`${message.role}-${index}`}
                  message={message}
                  conversationId={currentId}
                />
              ))}
              {stream ? <StreamingAssistant stream={stream} conversationId={currentId} /> : null}
            </>
          )}
        </section>

        <form className="composer" onSubmit={submitMessage}>
          <div className="composer-row">
            <textarea
              ref={inputRef}
              id="input"
              rows={1}
              value={input}
              placeholder="Message the MLSim assistant..."
              autoComplete="off"
              onChange={(event) => updateInput(event.target.value)}
              onKeyDown={handleTextareaKey}
            />
            {streaming ? (
              <button type="button" className="stop" aria-label="Stop" onClick={stopTurn}>
                <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
                  <rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor" />
                </svg>
              </button>
            ) : null}
            <button type="submit" className="send" aria-label="Send" disabled={streaming}>
              <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
                <path d="M4 12l15-7-7 15-2-6-6-2z" fill="currentColor" />
              </svg>
            </button>
          </div>
          <div className="composer-meta">
            <label
              className="sandbox"
              title="How much this turn is allowed to do in the copied Docker workspace"
            >
              <span className={`sandbox-ico ${sandboxClass(sandbox)}`} />
              <select
                id="sandbox"
                value={sandbox}
                onChange={(event) => setSandbox(event.target.value as SandboxMode)}
              >
                {sandboxModes.map((mode) => (
                  <option key={mode} value={mode}>
                    {sandboxLabel(mode)}
                  </option>
                ))}
              </select>
            </label>
            <label
              className="mode-toggle"
              title="Use autonomous AGENTS.md so the orchestrator proceeds with assumptions instead of asking clarification questions"
            >
              <input
                type="checkbox"
                checked={autonomous}
                onChange={(event) => setAutonomous(event.target.checked)}
              />
              <span className="toggle-track" aria-hidden="true">
                <span className="toggle-thumb" />
              </span>
              <span>Autonomous</span>
            </label>
            <span className="hint">Enter to send · Shift+Enter for newline</span>
          </div>
        </form>
      </main>
    </div>
  );
}

function Sidebar({
  conversations,
  currentId,
  onNewConversation,
  onSelect,
  onDelete,
}: {
  conversations: ConversationSummary[];
  currentId: string | null;
  onNewConversation: () => void | Promise<void>;
  onSelect: (id: string) => void | Promise<void>;
  onDelete: (id: string) => void | Promise<void>;
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <LogoMark />
        </div>
        <div className="brand-text">
          <h1>MLSim</h1>
          <p>assistant</p>
        </div>
      </div>

      <button className="new-chat" onClick={() => void onNewConversation()}>
        <span className="plus">+</span> New conversation
      </button>

      <nav className="conv-scroll">
        <ul className="conv-list">
          {conversations.map((conversation) => (
            <li
              key={conversation.id}
              className={`conv-item ${conversation.id === currentId ? "active" : ""}`}
            >
              <button
                className="conv-select"
                type="button"
                onClick={() => void onSelect(conversation.id)}
              >
                <span className="tick" />
                <span className="label">{conversation.title || "New chat"}</span>
              </button>
              <button
                className="del"
                type="button"
                title="Delete"
                onClick={() => void onDelete(conversation.id)}
              >
                x
              </button>
            </li>
          ))}
        </ul>
        {!conversations.length ? <p className="conv-empty">No conversations yet.</p> : null}
      </nav>

      <footer className="side-foot">
        <span className="dot" /> docker · codex · <code>gpt-5.3-codex-spark</code>
      </footer>
    </aside>
  );
}

function Welcome({ onSuggestion }: { onSuggestion: (text: string) => void }) {
  return (
    <div className="welcome">
      <h2>Ask MLSim anything</h2>
      <p>
        Ask about MLSim or delegate changes. Each conversation edits an isolated copy of{" "}
        <code>main/</code>, mounted into Docker for Codex.
      </p>
      <div className="chips">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion.fill}
            className="chip"
            type="button"
            onClick={() => onSuggestion(suggestion.fill)}
          >
            {suggestion.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageRow({
  message,
  conversationId,
}: {
  message: ChatMessage;
  conversationId: string | null;
}) {
  const isAssistant = message.role === "assistant";
  return (
    <div className={`msg ${isAssistant ? "assistant" : "user"}`}>
      <div className={`avatar ${isAssistant ? "assistant-avatar" : ""}`}>
        {isAssistant ? <LogoMark compact /> : "you"}
      </div>
      <div className="bubble">
        {isAssistant ? (
          <>
            <IntermediateOutputs outputs={message.intermediate_outputs || []} />
            <MarkdownBlock source={message.content} conversationId={conversationId} />
          </>
        ) : (
          message.content
        )}
      </div>
    </div>
  );
}

function StreamingAssistant({
  stream,
  conversationId,
}: {
  stream: StreamState;
  conversationId: string | null;
}) {
  const hasFinal = Boolean(stream.final);
  return (
    <div className="msg assistant">
      <div className="avatar assistant-avatar">
        <LogoMark compact />
      </div>
      <div className="bubble">
        <IntermediateOutputs outputs={stream.intermediateOutputs} />
        {!hasFinal && !stream.stopped ? (
          <div className="turn-work">
            {stream.implementer ? (
              <div className="role-output implementer">
                <div className="role-title">Implementer Summary</div>
                <MarkdownBlock source={stream.implementer} conversationId={conversationId} className="role-md" />
              </div>
            ) : null}
            <div className="progress">
              <div className="dots">
                <span />
                <span />
                <span />
              </div>
              <div className="line cur">{stream.progress}</div>
            </div>
          </div>
        ) : null}
        {stream.stopped ? <div className="stop-note">Stopped.</div> : null}
        {hasFinal ? <MarkdownBlock source={stream.final} conversationId={conversationId} /> : null}
      </div>
    </div>
  );
}

function IntermediateOutputs({ outputs }: { outputs: IntermediateOutput[] }) {
  const normalized = useMemo(
    () => outputs.filter((output) => output && output.text),
    [outputs],
  );
  if (!normalized.length) {
    return null;
  }
  return (
    <div className="intermediate-outputs">
      <div className="role-title">Intermediate Output</div>
      <div className="intermediate-output-list">
        {normalized.map((output, index) => (
          <div key={`${output.role || "note"}-${index}`} className="intermediate-output">
            {output.role ? <span className="intermediate-output-role">{output.role}</span> : null}
            <div className="intermediate-output-text">{normalizeBackendText(output.text)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MarkdownBlock({
  source,
  conversationId,
  className = "final-md",
}: {
  source: string;
  conversationId: string | null;
  className?: string;
}) {
  const html = useMemo(
    () => markdownHtml(source, conversationId),
    [source, conversationId],
  );
  return <div className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

function LogoMark({ compact = false }: { compact?: boolean }) {
  return (
    <svg
      className={`logo-glyph ${compact ? "compact" : ""}`}
      viewBox="0 0 44 44"
      fill="none"
      aria-hidden="true"
    >
      <path className="logo-shell" d="M22 5.5 36.5 13.75v16.5L22 38.5 7.5 30.25v-16.5L22 5.5Z" />
      <path className="logo-trace" d="M13.5 25.5 18.8 18l5.4 9.8 4.1-12.2 3.5 8.7h4.5" />
      <circle className="logo-node" cx="13.5" cy="25.5" r="2.2" />
      <circle className="logo-node" cx="24.2" cy="27.8" r="2.2" />
      <circle className="logo-node" cx="31.8" cy="24.3" r="2.2" />
    </svg>
  );
}

function sandboxClass(mode: string): string {
  if (mode === "read-only") {
    return "ro";
  }
  if (mode === "danger-full-access") {
    return "full";
  }
  return "";
}

function sandboxLabel(mode: string): string {
  if (mode === "read-only") {
    return "Read-only · ask / notify only";
  }
  if (mode === "danger-full-access") {
    return "Full access · bypass sandbox";
  }
  if (mode === "workspace-write") {
    return "Workspace-write · implement in copy";
  }
  return mode;
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
