import React, {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createRoot } from "react-dom/client";
import { Cube, PaperPlaneRight, Plus, Stop, Trash } from "@phosphor-icons/react";

import {
  DEFAULT_SANDBOX,
  cancelTurn,
  createConversation,
  deleteConversation,
  getConversation,
  listCodexBackends,
  listConversations,
  resumeTurn,
  streamTurn,
  updateConversationRuntime,
} from "./api";
import { cn } from "./lib/cn";
import { LegacyAssistant, TurnTimeline } from "./components/Turn";
import { markdownHtml } from "./markdown";
import { reduceTurn } from "./turn";
import type {
  ChatMessage,
  CodexModelOption,
  CodexRoleRuntime,
  CodexRuntimeSelection,
  Conversation,
  ConversationSummary,
  SandboxMode,
  TurnEvent,
} from "./types";

import "./index.css";

interface LiveTurn {
  events: TurnEvent[];
  toolCall: string;
  stopped: boolean;
}

interface PendingScrollRestore {
  scrollHeight: number;
  scrollTop: number;
}

const EMPTY_LIVE: LiveTurn = { events: [], toolCall: "", stopped: false };

const SUGGESTIONS = [
  {
    label: "List the L1 profilers",
    fill: "List the available VibeSim L1 profilers.",
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

/** `Sol·xhigh` — short enough for the runtime line under the composer. */
function runtimeSummary(runtime: CodexRoleRuntime): string {
  const name = runtime.model.includes("DeepSeek")
    ? "DS"
    : runtime.model.replace(/^gpt-5\.6-/i, "").replace(/^gpt-/i, "");
  const tier = runtime.serviceTier === "fast" ? "\u00b7fast" : "";
  return runtime.effort ? `${name}\u00b7${runtime.effort}${tier}` : `${name}${tier}`;
}

function App() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [title, setTitle] = useState("VibeSim Assistant");
  const [sandboxModes, setSandboxModes] = useState<SandboxMode[]>([
    "read-only",
    "workspace-write",
    "danger-full-access",
  ]);
  const [sandbox, setSandbox] = useState<SandboxMode>(
    (localStorage.getItem("vibesim_sandbox") as SandboxMode | null) || DEFAULT_SANDBOX,
  );
  const [autonomous, setAutonomous] = useState(
    localStorage.getItem("vibesim_autonomous") === "1",
  );
  const [modelOptions, setModelOptions] = useState<CodexModelOption[]>([]);
  // Replaced by the server catalog's defaults as soon as it answers.
  const [codexRuntime, setCodexRuntime] = useState<CodexRuntimeSelection>({
    orchestrator: { model: "", effort: "", serviceTier: "default" },
    implementer: { model: "", effort: "", serviceTier: "default" },
  });
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [live, setLive] = useState<LiveTurn | null>(null);
  const [messageStartIndex, setMessageStartIndex] = useState(0);
  const [hasOlderMessages, setHasOlderMessages] = useState(false);
  const [loadingOlderMessages, setLoadingOlderMessages] = useState(false);
  const [olderMessagesError, setOlderMessagesError] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const messagesRef = useRef<HTMLElement | null>(null);
  const olderMessagesSentinelRef = useRef<HTMLDivElement | null>(null);
  const loadingOlderMessagesRef = useRef(false);
  const olderMessagesRequestSequenceRef = useRef(0);
  const selectedConversationIdRef = useRef<string | null>(null);
  const pendingScrollRestoreRef = useRef<PendingScrollRestore | null>(null);
  // Follow streaming output only while the reader stays near the bottom.
  // Once they scroll up to inspect history, live events must not steal position.
  const shouldAutoScrollRef = useRef(true);

  useEffect(() => {
    void (async () => {
      const catalog = await listCodexBackends();
      setModelOptions(catalog.models);
      setCodexRuntime(catalog.defaults);
      const items = await refreshSidebar();
      if (items.length) await selectConversation(items[0].id);
    })();
  }, []);

  useEffect(() => {
    localStorage.setItem("vibesim_sandbox", sandbox);
  }, [sandbox]);

  useEffect(() => {
    localStorage.setItem("vibesim_autonomous", autonomous ? "1" : "0");
  }, [autonomous]);

  useEffect(() => {
    if (!shouldAutoScrollRef.current) {
      return;
    }
    const animationFrame = requestAnimationFrame(() => {
      const messagesElement = messagesRef.current;
      if (messagesElement && shouldAutoScrollRef.current) {
        messagesElement.scrollTop = messagesElement.scrollHeight;
      }
    });
    return () => cancelAnimationFrame(animationFrame);
  }, [messages, live]);

  useLayoutEffect(() => {
    const restore = pendingScrollRestoreRef.current;
    const messagesElement = messagesRef.current;
    if (!restore || !messagesElement) {
      return;
    }
    messagesElement.scrollTop =
      restore.scrollTop + (messagesElement.scrollHeight - restore.scrollHeight);
    pendingScrollRestoreRef.current = null;
  }, [messages]);

  useEffect(() => {
    const messagesElement = messagesRef.current;
    const sentinelElement = olderMessagesSentinelRef.current;
    if (
      !messagesElement ||
      !sentinelElement ||
      !currentId ||
      !hasOlderMessages ||
      streaming ||
      olderMessagesError
    ) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          void loadOlderMessages();
        }
      },
      {
        root: messagesElement,
        // Start fetching shortly before the reader reaches the exact top.
        rootMargin: "160px 0px 0px",
      },
    );
    observer.observe(sentinelElement);
    return () => observer.disconnect();
  }, [currentId, hasOlderMessages, messageStartIndex, olderMessagesError, streaming]);

  function handleMessagesScroll(): void {
    const messagesElement = messagesRef.current;
    if (!messagesElement) {
      return;
    }
    const distanceFromBottom =
      messagesElement.scrollHeight - messagesElement.scrollTop - messagesElement.clientHeight;
    shouldAutoScrollRef.current = distanceFromBottom <= 48;
  }

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
    shouldAutoScrollRef.current = true;
    selectedConversationIdRef.current = id;
    const conversation = await getConversation(id);
    if (!conversation) {
      await refreshSidebar();
      return;
    }
    loadConversation(conversation);
    void reconnectTurn(id);
  }

  function loadConversation(conversation: Conversation): void {
    olderMessagesRequestSequenceRef.current += 1;
    loadingOlderMessagesRef.current = false;
    pendingScrollRestoreRef.current = null;
    selectedConversationIdRef.current = conversation.id;
    setCurrentId(conversation.id);
    setMessages(conversation.messages || []);
    setMessageStartIndex(conversation.message_page?.start_index || 0);
    setHasOlderMessages(Boolean(conversation.message_page?.has_more));
    setLoadingOlderMessages(false);
    setOlderMessagesError("");
    if (conversation.sandbox) {
      setSandbox(conversation.sandbox as SandboxMode);
    }
    setAutonomous(Boolean(conversation.autonomous));
    if (conversation.codex_runtime) {
      setCodexRuntime(conversation.codex_runtime);
    }
    setTitle(
      conversation.title && conversation.title !== "New chat"
        ? conversation.title
        : "VibeSim Assistant",
    );
  }

  async function loadOlderMessages(): Promise<void> {
    const conversationId = currentId;
    if (
      !conversationId ||
      !hasOlderMessages ||
      streaming ||
      loadingOlderMessagesRef.current
    ) {
      return;
    }

    loadingOlderMessagesRef.current = true;
    const requestSequence = ++olderMessagesRequestSequenceRef.current;
    setLoadingOlderMessages(true);
    setOlderMessagesError("");
    try {
      const olderConversation = await getConversation(conversationId, messageStartIndex);
      if (
        !olderConversation ||
        selectedConversationIdRef.current !== conversationId ||
        olderMessagesRequestSequenceRef.current !== requestSequence
      ) {
        return;
      }
      const olderPage = olderConversation.message_page;
      if (!olderPage || olderPage.end_index !== messageStartIndex) {
        throw new Error("backend returned a non-contiguous history page");
      }

      const messagesElement = messagesRef.current;
      if (messagesElement) {
        pendingScrollRestoreRef.current = {
          scrollHeight: messagesElement.scrollHeight,
          scrollTop: messagesElement.scrollTop,
        };
      }
      shouldAutoScrollRef.current = false;
      setMessages((currentMessages) => [
        ...(olderConversation.messages || []),
        ...currentMessages,
      ]);
      setMessageStartIndex(olderPage.start_index);
      setHasOlderMessages(olderPage.has_more);
    } catch (error) {
      if (olderMessagesRequestSequenceRef.current === requestSequence) {
        setOlderMessagesError(`Could not load older messages: ${String(error)}`);
      }
    } finally {
      if (olderMessagesRequestSequenceRef.current === requestSequence) {
        loadingOlderMessagesRef.current = false;
        setLoadingOlderMessages(false);
      }
    }
  }

  async function newConversation(): Promise<void> {
    if (streaming) {
      return;
    }
    const conversation = await createConversation(sandbox, autonomous, codexRuntime);
    shouldAutoScrollRef.current = true;
    olderMessagesRequestSequenceRef.current += 1;
    loadingOlderMessagesRef.current = false;
    pendingScrollRestoreRef.current = null;
    selectedConversationIdRef.current = conversation.id;
    setCurrentId(conversation.id);
    setMessages([]);
    setMessageStartIndex(0);
    setHasOlderMessages(false);
    setLoadingOlderMessages(false);
    setOlderMessagesError("");
    setTitle("VibeSim Assistant");
    await refreshSidebar();
    inputRef.current?.focus();
  }

  async function removeConversation(id: string): Promise<void> {
    if (streaming) {
      return;
    }
    await deleteConversation(id);
    if (currentId === id) {
      olderMessagesRequestSequenceRef.current += 1;
      loadingOlderMessagesRef.current = false;
      pendingScrollRestoreRef.current = null;
      selectedConversationIdRef.current = null;
      setCurrentId(null);
      setMessages([]);
      setMessageStartIndex(0);
      setHasOlderMessages(false);
      setLoadingOlderMessages(false);
      setOlderMessagesError("");
      setTitle("VibeSim Assistant");
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
      const conversation = await createConversation(sandbox, autonomous, codexRuntime);
      conversationId = conversation.id;
      setCurrentId(conversation.id);
      await refreshSidebar();
    } else {
      // Effort, and a sibling model, stay changeable mid-conversation; the
      // server rejects only a family change once the conversation has history.
      await updateConversationRuntime(conversationId, codexRuntime);
    }

    setInput("");
    shouldAutoScrollRef.current = true;
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
    }
    setMessages((current) => [...current, { role: "user", content: text }]);
    setLive({ ...EMPTY_LIVE });
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
        liveTurnHandlers(),
        controller.signal,
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setLive((current) => ({ ...(current || EMPTY_LIVE), stopped: true }));
      } else {
        localOnlyAssistant = {
          role: "assistant",
          content: `(error talking to backend: ${String(error)})`,
        };
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
      setLive(null);
      await refreshSidebar();
      inputRef.current?.focus();
    }
  }

  function liveTurnHandlers() {
    return {
      toolCall: (line: string) => {
        if (!line) {
          return;
        }
        setLive((current) => ({ ...(current || EMPTY_LIVE), toolCall: line }));
      },
      event: (turnEvent: TurnEvent) => {
        setLive((current) => ({
          ...(current || EMPTY_LIVE),
          events: [...(current?.events || []), turnEvent],
        }));
      },
    };
  }

  async function reconnectTurn(conversationId: string): Promise<void> {
    const controller = new AbortController();
    abortRef.current = controller;
    setLive({ ...EMPTY_LIVE });
    setStreaming(true);
    try {
      const attached = await resumeTurn(
        conversationId,
        liveTurnHandlers(),
        controller.signal,
      );
      if (!attached) {
        return;
      }
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        setLive((current) => ({
          ...(current || EMPTY_LIVE),
          toolCall: `(error reconnecting to backend: ${String(error)})`,
        }));
      }
    } finally {
      if (abortRef.current !== controller) {
        return;
      }
      setStreaming(false);
      abortRef.current = null;
      const conversation = await getConversation(conversationId);
      if (conversation) {
        loadConversation(conversation);
      }
      setLive(null);
      await refreshSidebar();
      inputRef.current?.focus();
    }
  }

  async function stopTurn(): Promise<void> {
    const conversationId = currentId;
    const controller = abortRef.current;
    if (!streaming || !conversationId || !controller) {
      return;
    }
    try {
      await cancelTurn(conversationId);
    } catch (error) {
      setLive((current) => ({
        ...(current || EMPTY_LIVE),
        toolCall: `(error stopping backend turn: ${String(error)})`,
      }));
    } finally {
      controller.abort();
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

  const isEmpty = messages.length === 0 && !live;

  return (
    <div className="flex h-full flex-col text-zinc-200">
      <header className="shrink-0 border-b border-hair bg-ink/85 backdrop-blur-md">
        <div className="flex w-full items-center gap-3 px-[5%] py-2.5">
          <span className="grid h-6 w-6 place-items-center rounded-md bg-gradient-to-br from-orch-soft to-orch text-ink">
            <Cube size={13} weight="bold" />
          </span>
          <span className="font-semibold text-zinc-100">{title}</span>
          <span className="hidden text-[13px] text-zinc-500 sm:inline">
            drives <code className="font-mono text-[12px] text-orch-soft">codex exec</code> in a
            copied <code className="font-mono text-[12px] text-orch-soft">main/</code> workspace
          </span>
        </div>
      </header>

      <div className="flex min-h-0 w-full flex-1 px-[5%]">
        <Sidebar
          conversations={conversations}
          currentId={currentId}
          streaming={streaming}
          onNewConversation={newConversation}
          onSelect={selectConversation}
          onDelete={removeConversation}
          backendSummary={`${runtimeSummary(codexRuntime.orchestrator)} / ${runtimeSummary(codexRuntime.implementer)}`}
        />

        <main className="flex min-w-0 flex-1 flex-col">
          <section
            ref={messagesRef}
            className="scroll flex-1 space-y-6 overflow-y-auto px-6 py-7"
            onScroll={handleMessagesScroll}
          >
            {isEmpty ? (
              <Welcome
                autonomous={autonomous}
                onAutonomousChange={setAutonomous}
                onSuggestion={(fill) => updateInput(fill)}
              />
            ) : (
              <>
                <div className="relative h-5 text-center font-mono text-[10.5px] text-zinc-600">
                  <div ref={olderMessagesSentinelRef} className="absolute inset-x-0 top-0 h-px" />
                  {loadingOlderMessages ? (
                    <span>loading earlier messages…</span>
                  ) : olderMessagesError ? (
                    <button
                      type="button"
                      className="text-orch-soft hover:underline"
                      onClick={() => void loadOlderMessages()}
                    >
                      retry loading earlier messages
                    </button>
                  ) : !hasOlderMessages ? (
                    <span>beginning of conversation</span>
                  ) : null}
                </div>
                {messages.map((message, index) =>
                  message.role === "user" ? (
                    <UserBubble
                      key={`${currentId}:${messageStartIndex + index}`}
                      content={message.content}
                      conversationId={currentId}
                    />
                  ) : (
                    <AssistantTurn
                      key={`${currentId}:${messageStartIndex + index}`}
                      message={message}
                      conversationId={currentId}
                    />
                  ),
                )}
                {live ? (
                  <StreamingTurn live={live} conversationId={currentId} streaming={streaming} />
                ) : null}
              </>
            )}
          </section>

          <div className="shrink-0 border-t border-hairsoft px-6 py-4">
            <form onSubmit={submitMessage}>
              <div className="flex items-end gap-2 rounded-2xl border border-hair bg-panel px-4 py-2.5 focus-within:border-orch-line">
                <textarea
                  ref={inputRef}
                  id="input"
                  rows={1}
                  value={input}
                  placeholder="Message the VibeSim assistant…"
                  autoComplete="off"
                  className="flex-1 resize-none bg-transparent py-1 text-[14.5px] leading-relaxed text-zinc-100 outline-none placeholder:text-zinc-500"
                  onChange={(nativeEvent) => updateInput(nativeEvent.target.value)}
                  onKeyDown={handleTextareaKey}
                />
                {streaming ? (
                  <button
                    type="button"
                    aria-label="Stop"
                    onClick={stopTurn}
                    className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-hair text-zinc-300 transition hover:text-zinc-100"
                  >
                    <Stop size={16} weight="fill" />
                  </button>
                ) : null}
                <button
                  type="submit"
                  aria-label="Send"
                  disabled={streaming || !input.trim()}
                  className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-orch-soft to-orch text-ink transition hover:brightness-110 disabled:opacity-40"
                >
                  <PaperPlaneRight size={16} weight="bold" />
                </button>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-3 text-[12.5px] text-zinc-500">
                {(["orchestrator", "implementer"] as const).map((role) => {
                  const selected = modelOptions.find(
                    (model) => model.id === codexRuntime[role].model,
                  );
                  // A started conversation keeps its Codex session, which only a
                  // model of the recording family can resume. Effort stays free.
                  const familyLocked = messages.length > 0;
                  const selectableModels = modelOptions.filter(
                    (model) => !familyLocked || model.family === selected?.family,
                  );
                  return (
                    <label key={role} className="flex items-center gap-1.5">
                      <span
                        className={role === "orchestrator" ? "text-orch-soft" : "text-impl-soft"}
                      >
                        {role === "orchestrator" ? "Orchestrator" : "Implementer"}
                      </span>
                      <select
                        aria-label={`${role} Codex model`}
                        value={codexRuntime[role].model}
                        disabled={streaming}
                        onChange={(nativeEvent) => {
                          const model = modelOptions.find(
                            (option) => option.id === nativeEvent.target.value,
                          );
                          if (!model) return;
                          setCodexRuntime((current) => ({
                            ...current,
                            [role]: {
                              model: model.id,
                              effort: model.efforts.includes(current[role].effort)
                                ? current[role].effort
                                : model.defaultEffort,
                              serviceTier: model.serviceTiers.includes(
                                current[role].serviceTier,
                              )
                                ? current[role].serviceTier
                                : model.defaultServiceTier,
                            },
                          }));
                        }}
                        className="cursor-pointer rounded-md border border-hair bg-panel px-2 py-1 text-[12.5px] text-zinc-300 outline-none disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {selectableModels.map((model) => (
                          <option key={model.id} value={model.id} disabled={!model.available}>
                            {model.label}
                            {model.available ? "" : " (unavailable)"}
                          </option>
                        ))}
                      </select>
                      <select
                        aria-label={`${role} reasoning effort`}
                        value={codexRuntime[role].effort}
                        disabled={streaming}
                        onChange={(nativeEvent) =>
                          setCodexRuntime((current) => ({
                            ...current,
                            [role]: { ...current[role], effort: nativeEvent.target.value },
                          }))
                        }
                        className="cursor-pointer rounded-md border border-hair bg-panel px-2 py-1 text-[12.5px] text-zinc-300 outline-none disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {(selected?.efforts ?? []).map((effort) => (
                          <option key={effort} value={effort}>
                            {effort}
                          </option>
                        ))}
                      </select>
                      <span
                        role="group"
                        aria-label={`${role} speed tier`}
                        className="inline-flex rounded-md border border-hair bg-panel p-0.5"
                      >
                        {(["default", "fast"] as const).map((serviceTier) => {
                          const supported = selected?.serviceTiers.includes(serviceTier) ?? false;
                          const active = codexRuntime[role].serviceTier === serviceTier;
                          return (
                            <button
                              key={serviceTier}
                              type="button"
                              aria-pressed={active}
                              aria-label={`${role} ${serviceTier === "fast" ? "Fast" : "Normal"} tier`}
                              disabled={streaming || !supported}
                              title={
                                serviceTier === "fast" && !supported
                                  ? "This model does not offer the Fast service tier"
                                  : undefined
                              }
                              onClick={() =>
                                setCodexRuntime((current) => ({
                                  ...current,
                                  [role]: { ...current[role], serviceTier },
                                }))
                              }
                              className={cn(
                                "rounded px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide transition disabled:cursor-not-allowed disabled:opacity-35",
                                active
                                  ? "bg-zinc-700 text-zinc-100 shadow-sm"
                                  : "text-zinc-500 hover:text-zinc-300",
                              )}
                            >
                              {serviceTier === "fast" ? "Fast" : "Normal"}
                            </button>
                          );
                        })}
                      </span>
                    </label>
                  );
                })}
                <label
                  className="flex cursor-pointer items-center gap-2"
                  title="How much this turn is allowed to do in the copied Docker workspace"
                >
                  <span className={cn("h-1.5 w-1.5 rounded-full", sandboxDotClass(sandbox))} />
                  <select
                    value={sandbox}
                    onChange={(nativeEvent) => setSandbox(nativeEvent.target.value as SandboxMode)}
                    className="cursor-pointer rounded-md border border-hair bg-panel px-2 py-1 text-[12.5px] text-zinc-300 outline-none"
                  >
                    {sandboxModes.map((mode) => (
                      <option key={mode} value={mode}>
                        {sandboxLabel(mode)}
                      </option>
                    ))}
                  </select>
                </label>
                <span className="ml-auto">Enter to send · Shift+Enter for newline</span>
              </div>
            </form>
          </div>
        </main>
      </div>
    </div>
  );
}

function Sidebar({
  conversations,
  currentId,
  streaming,
  onNewConversation,
  onSelect,
  onDelete,
  backendSummary,
}: {
  conversations: ConversationSummary[];
  currentId: string | null;
  streaming: boolean;
  onNewConversation: () => void | Promise<void>;
  onSelect: (id: string) => void | Promise<void>;
  onDelete: (id: string) => void | Promise<void>;
  backendSummary: string;
}) {
  return (
    <aside className="hidden w-[230px] shrink-0 flex-col gap-1 border-r border-hairsoft py-4 pr-4 md:flex">
      <button
        onClick={() => void onNewConversation()}
        disabled={streaming}
        className="mb-3 flex items-center justify-center gap-2 rounded-lg border border-orch-line bg-orch-bg px-3 py-2 text-[13.5px] font-semibold text-orch-soft transition hover:brightness-110 disabled:opacity-40"
      >
        <Plus size={14} weight="bold" /> New conversation
      </button>

      <nav className="scroll -mr-1 flex-1 space-y-1 overflow-y-auto pr-1">
        {conversations.map((conversation) => {
          const active = conversation.id === currentId;
          return (
            <div
              key={conversation.id}
              className={cn(
                "group flex items-center gap-1 rounded-lg px-1 text-[13px]",
                active ? "border border-hair bg-panel2" : "border border-transparent hover:bg-panel",
              )}
            >
              <button
                type="button"
                onClick={() => void onSelect(conversation.id)}
                className="flex min-w-0 flex-1 items-center gap-2 py-2 pl-2 text-left"
              >
                <span
                  className={cn(
                    "h-1.5 w-1.5 shrink-0 rounded-full",
                    active ? "bg-orch" : "bg-zinc-600",
                  )}
                />
                <span className="truncate text-zinc-300">{conversation.title || "New chat"}</span>
              </button>
              <button
                type="button"
                title="Delete"
                onClick={() => void onDelete(conversation.id)}
                className="grid h-6 w-6 shrink-0 place-items-center rounded-md text-zinc-600 opacity-0 transition hover:text-zinc-300 group-hover:opacity-100"
              >
                <Trash size={13} />
              </button>
            </div>
          );
        })}
        {!conversations.length ? (
          <p className="px-2 py-2 text-[12.5px] text-zinc-600">No conversations yet.</p>
        ) : null}
      </nav>

      <footer className="mt-auto flex items-center gap-2 border-t border-hairsoft pt-3 text-[12px] text-zinc-500">
        <span className="h-1.5 w-1.5 rounded-full bg-impl" /> docker · codex ·{" "}
        <span className="font-mono text-[11px]">{backendSummary}</span>
      </footer>
    </aside>
  );
}

function Welcome({
  autonomous,
  onAutonomousChange,
  onSuggestion,
}: {
  autonomous: boolean;
  onAutonomousChange: (enabled: boolean) => void;
  onSuggestion: (text: string) => void;
}) {
  return (
    <div className="mx-auto mt-[8vh] max-w-[640px] text-center">
      <span className="mx-auto mb-4 grid h-11 w-11 place-items-center rounded-xl bg-gradient-to-br from-orch-soft to-orch text-ink">
        <Cube size={22} weight="bold" />
      </span>
      <h2 className="text-[22px] font-semibold text-zinc-100">Ask VibeSim anything</h2>
      <p className="mx-auto mt-2 max-w-[520px] text-[14px] leading-relaxed text-zinc-400">
        Ask about VibeSim or delegate changes. Each conversation edits an isolated copy of{" "}
        <code className="font-mono text-[13px] text-orch-soft">main/</code>, mounted into Docker for
        Codex.
      </p>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion.fill}
            type="button"
            onClick={() => onSuggestion(suggestion.fill)}
            className="rounded-full border border-hair bg-panel px-3.5 py-1.5 text-[13px] text-zinc-300 transition hover:border-orch-line hover:text-orch-soft"
          >
            {suggestion.label}
          </button>
        ))}
      </div>
      <button
        type="button"
        aria-pressed={autonomous}
        title="Use autonomous AGENTS.md so the orchestrator proceeds with assumptions instead of asking clarification questions"
        onClick={() => onAutonomousChange(!autonomous)}
        className={cn(
          "mt-6 inline-flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-[12.5px] transition",
          autonomous
            ? "border-impl-line bg-impl-bg text-impl-soft"
            : "border-hair bg-panel text-zinc-400 hover:text-zinc-200",
        )}
      >
        <span
          className={cn("h-1.5 w-1.5 rounded-full", autonomous ? "bg-impl" : "bg-zinc-600")}
        />
        {autonomous ? "Autonomous on" : "Autonomous off"}
      </button>
    </div>
  );
}

function UserBubble({
  content,
  conversationId,
}: {
  content: string;
  conversationId: string | null;
}) {
  const html = useMemo(() => markdownHtml(content, conversationId), [content, conversationId]);
  return (
    <div className="flex justify-end">
      <div
        className="prose-msg ml-auto max-w-[78%] rounded-2xl rounded-br-md border border-hair bg-panel2 px-4 py-2.5 text-[14.5px] leading-relaxed"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}

function AssistantTurn({
  message,
  conversationId,
}: {
  message: ChatMessage;
  conversationId: string | null;
}) {
  const cards = useMemo(
    () => (message.activity && message.activity.length ? reduceTurn(message.activity) : null),
    [message.activity],
  );
  if (cards) {
    return <TurnTimeline cards={cards} conversationId={conversationId} />;
  }
  return (
    <LegacyAssistant
      content={message.content}
      intermediateOutputs={message.intermediate_outputs}
      conversationId={conversationId}
      failure={message.failure}
    />
  );
}

function StreamingTurn({
  live,
  conversationId,
  streaming,
}: {
  live: LiveTurn;
  conversationId: string | null;
  streaming: boolean;
}) {
  const cards = useMemo(() => reduceTurn(live.events), [live.events]);
  if (live.stopped) {
    return (
      <div className="animate-rise w-full space-y-2.5">
        <TurnTimeline cards={cards} conversationId={conversationId} />
        <div className="rounded-xl border border-hair bg-panel/80 px-3.5 py-2.5 text-[13px] text-zinc-500">
          Stopped.
        </div>
      </div>
    );
  }
  return (
    <TurnTimeline
      cards={cards}
      streaming={streaming}
      toolCall={live.toolCall}
      conversationId={conversationId}
    />
  );
}

function sandboxDotClass(mode: string): string {
  if (mode === "read-only") {
    return "bg-impl";
  }
  if (mode === "danger-full-access") {
    return "bg-red-400";
  }
  return "bg-orch";
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
