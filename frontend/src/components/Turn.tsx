import { useMemo, type ReactNode } from "react";
import {
  Check,
  CheckCircle,
  Cpu,
  Database,
  FlagCheckered,
  GearSix,
  PaperPlaneRight,
  Prohibit,
  Question,
  Sparkle,
  Warning,
} from "@phosphor-icons/react";

import { cn } from "@/lib/cn";
import { markdownHtml, normalizeBackendText } from "@/markdown";
import { formatDuration, formatTokens, stripRolePrefix } from "@/turn";
import type {
  IntermediateOutput,
  Role,
  RoleNote,
  RolePhase,
  TerminalOutcome,
  Tokens,
  TurnCard,
  TurnFailure,
} from "@/types";
import { ROLE_STYLE } from "./roleStyles";

/** Which model actually ran this phase, shrunk for the card subtitle. */
function phaseRuntimeLabel(phase: RolePhase): string {
  const model = phase.runtime?.model;
  if (!model) return "codex";
  const name = model.includes("DeepSeek")
    ? "DeepSeek"
    : model.replace(/^gpt-5\.6-/i, "").replace(/^gpt-/i, "");
  return phase.runtime?.effort ? `${name}\u00b7${phase.runtime.effort}` : name;
}

function LoadingDots({ className }: { className: string }) {
  return (
    <span className="flex gap-0.5">
      <span className={cn("h-1 w-1 animate-pulse2 rounded-full", className)} />
      <span className={cn("h-1 w-1 animate-pulse2 rounded-full [animation-delay:.2s]", className)} />
      <span className={cn("h-1 w-1 animate-pulse2 rounded-full [animation-delay:.4s]", className)} />
    </span>
  );
}

function ToolCallLine({ text, dotClass, textClass }: { text: string; dotClass: string; textClass: string }) {
  return (
    <div className={cn("mt-3 flex items-center gap-2 font-mono text-[12px]", textClass)}>
      <LoadingDots className={dotClass} />
      <span className="truncate">{text || "working…"}</span>
    </div>
  );
}

function NoteList({ notes, dotClass }: { notes: RoleNote[]; dotClass: string }) {
  if (!notes.length) {
    return null;
  }
  return (
    <ul className="space-y-1">
      {notes.map((note, index) => (
        <li
          key={index}
          className={cn(
            "animate-noteIn flex gap-2 text-[13px] leading-snug text-zinc-300",
            note.level === "milestone" &&
              "rounded-md border border-emerald-400/15 bg-emerald-400/[0.045] px-2 py-1.5 text-zinc-200",
          )}
        >
          {note.level === "milestone" ? (
            <FlagCheckered className="mt-0.5 shrink-0 text-emerald-300/75" size={12} weight="fill" />
          ) : (
            <span className={cn("mt-[7px] h-1 w-1 shrink-0 rounded-full", dotClass)} />
          )}
          <span>{normalizeBackendText(note.text)}</span>
        </li>
      ))}
    </ul>
  );
}

function TokenStat({ icon, value, label, title }: { icon: ReactNode; value: string; label: string; title: string }) {
  return (
    <span className="flex items-center gap-1.5" title={title}>
      {icon}
      <span className="text-zinc-300">{value}</span> {label}
    </span>
  );
}

function TokenFooter({ tokens }: { tokens: Tokens }) {
  return (
    <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-hairsoft pt-2.5 font-mono text-[10.5px] text-zinc-500">
      <TokenStat
        icon={<Database size={12} className="text-zinc-600" />}
        value={formatTokens(tokens.read)}
        label="prefix read"
        title="cached input tokens (prefix cache hit)"
      />
      <TokenStat
        icon={<Cpu size={12} className="text-zinc-600" />}
        value={formatTokens(tokens.prefill)}
        label="prefill"
        title="uncached input tokens (cache write)"
      />
      <TokenStat
        icon={<Sparkle size={12} className="text-zinc-600" />}
        value={formatTokens(tokens.output)}
        label="output"
        title="generated output tokens"
      />
    </div>
  );
}

function MarkdownBody({ source, conversationId }: { source: string; conversationId: string | null }) {
  const html = useMemo(() => markdownHtml(source, conversationId), [source, conversationId]);
  return (
    <div
      className="prose-msg text-[14px] leading-relaxed"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

interface CardProps {
  avatar: ReactNode;
  avatarClass: string;
  title: string;
  sub?: string;
  chip?: ReactNode;
  working?: boolean;
  workingClass?: string;
  /** quiet left-accent bar (2px, low opacity) keyed to card type */
  accent?: string;
  /** faint flow-tinted card background */
  tint?: string;
  bodyLabel?: string;
  children?: ReactNode;
}

/** One uniform card, reused for role phases, hand-offs, and the answer. */
function Card({ avatar, avatarClass, title, sub, chip, working, workingClass, accent, tint, bodyLabel, children }: CardProps) {
  return (
    <div
      className={cn(
        "animate-grow rounded-xl border p-3.5",
        tint || "bg-panel/80",
        working ? workingClass : "border-hair",
        accent,
      )}
    >
      <div className="flex items-center gap-2.5">
        <span
          className={cn(
            "grid h-7 w-7 shrink-0 place-items-center rounded-lg border",
            avatarClass,
          )}
        >
          {avatar}
        </span>
        <span className="text-[13.5px] font-semibold text-zinc-100">{title}</span>
        {sub ? <span className="text-[12.5px] text-zinc-500">· {sub}</span> : null}
        {chip ? <span className="ml-auto">{chip}</span> : null}
      </div>
      {bodyLabel ? (
        <div className="mb-1.5 mt-2.5 font-mono text-[10px] uppercase tracking-wide text-zinc-600">
          {bodyLabel}
        </div>
      ) : (
        <div className="mt-2.5" />
      )}
      {children}
    </div>
  );
}

function RoleCard({
  phase,
  working,
  toolCall,
}: {
  phase: RolePhase;
  working: boolean;
  toolCall: string;
}) {
  const style = ROLE_STYLE[phase.role];
  const RoleIcon = style.Icon;
  const duration = formatDuration(phase.durationMs);
  const chip = working ? (
    <span
      className={cn(
        "flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[10.5px]",
        style.chipWorking,
      )}
    >
      <span className={cn("h-1.5 w-1.5 animate-pulse2 rounded-full", style.dot)} /> working
    </span>
  ) : (
    <span className="flex items-center gap-1.5 rounded-full border border-hair px-2 py-0.5 font-mono text-[10.5px] text-zinc-400">
      <Check size={10} weight="fill" className="text-zinc-500" />
      {duration || "done"}
    </span>
  );

  return (
    <Card
      avatar={<RoleIcon size={13} weight="fill" />}
      avatarClass={style.avatar}
      title={style.label}
      sub={`${phaseRuntimeLabel(phase)} · round ${phase.round}`}
      chip={chip}
      working={working}
      workingClass={style.cardWorking}
      accent={style.accent}
      tint={style.tint}
      bodyLabel={`intermediate output · ${phase.notes.length}`}
    >
      <NoteList notes={phase.notes} dotClass={style.dot} />
      {working ? (
        <ToolCallLine
          text={stripRolePrefix(toolCall)}
          dotClass={style.dot}
          textClass={style.progressText}
        />
      ) : null}
      {phase.tokens ? <TokenFooter tokens={phase.tokens} /> : null}
    </Card>
  );
}

function HandoffCard({
  variant,
  text,
  conversationId,
}: {
  variant: "delegated_task" | "conclusion";
  text: string;
  conversationId: string | null;
}) {
  const isTask = variant === "delegated_task";
  const style = ROLE_STYLE[isTask ? "orchestrator" : "implementer"];
  return (
    <Card
      avatar={
        isTask ? (
          <PaperPlaneRight size={13} weight="fill" />
        ) : (
          <FlagCheckered size={13} weight="fill" />
        )
      }
      avatarClass={style.avatar}
      title={isTask ? "Orchestrator → Implementer" : "Implementer → Orchestrator"}
      accent={isTask ? "border-l-2 border-l-orch/30" : "border-l-2 border-l-impl/30"}
      tint={isTask ? "tint-orch" : "tint-impl"}
      bodyLabel={isTask ? "delegated task" : "conclusion"}
    >
      <MarkdownBody source={text} conversationId={conversationId} />
    </Card>
  );
}

function TerminalResponseCard({
  text,
  outcome = "final_answer",
  role = "orchestrator",
  conversationId,
}: {
  text: string;
  outcome?: TerminalOutcome;
  /** The asking role. Only "Input needed" is role-tinted; an answer is an answer. */
  role?: Role;
  conversationId: string | null;
}) {
  // Stopped is deliberately neither of the other two: not the answer's green,
  // which would claim the agent replied, and not the failure's red, which would
  // claim something went wrong. The user ended this turn on purpose.
  if (outcome === "cancelled") {
    return (
      <Card
        avatar={<Prohibit size={13} weight="bold" />}
        avatarClass="border-hair bg-panel2 text-zinc-400"
        title="Stopped"
        accent="border-l-2 border-l-white/15"
        bodyLabel="interrupted"
      >
        <MarkdownBody source={text} conversationId={conversationId} />
      </Card>
    );
  }
  const needsInput = outcome === "request_user_input";
  const asking = ROLE_STYLE[role];
  return (
    <Card
      avatar={needsInput ? <Question size={13} weight="bold" /> : <CheckCircle size={13} weight="fill" />}
      avatarClass={needsInput ? asking.avatar : "border-ans-line bg-ans-bg text-ans"}
      title={needsInput ? "Input needed" : "Answer"}
      accent={needsInput ? asking.accent : "border-l-2 border-l-ans/70"}
      tint={needsInput ? asking.tint : "tint-answer"}
      bodyLabel={needsInput ? "clarification" : "final answer"}
    >
      <MarkdownBody source={text} conversationId={conversationId} />
    </Card>
  );
}

/**
 * A turn that ended without an answer. Deliberately not styled as a response:
 * a failure rendered in the answer card reads as though the agent replied.
 */
function FailureCard({
  text,
  code,
  conversationId,
}: {
  text: string;
  code?: string;
  conversationId: string | null;
}) {
  return (
    <Card
      avatar={<Warning size={13} weight="fill" />}
      avatarClass="border-fail-line bg-fail-bg text-fail"
      title="Turn failed"
      accent="border-l-2 border-l-fail/70"
      tint="tint-fail"
      bodyLabel={code || "error"}
    >
      <MarkdownBody source={text} conversationId={conversationId} />
    </Card>
  );
}

function PreparingCard({ toolCall }: { toolCall: string }) {
  return (
    <Card
      avatar={<GearSix size={13} weight="fill" />}
      avatarClass="border-hair bg-panel2 text-zinc-400"
      title="Preparing workspace"
      working
      workingClass="border-hair ring-1 ring-white/5"
      chip={
        <span className="flex items-center gap-1.5 rounded-full border border-hair px-2 py-0.5 font-mono text-[10.5px] text-zinc-400">
          <span className="h-1.5 w-1.5 animate-pulse2 rounded-full bg-zinc-400" /> working
        </span>
      }
    >
      <ToolCallLine text={toolCall} dotClass="bg-zinc-500" textClass="text-zinc-400" />
    </Card>
  );
}

/**
 * The full assistant turn as a role timeline. `streaming` marks the last open
 * role phase as working (and surfaces the transient `toolCall` line); on reload
 * everything is settled.
 */
export function TurnTimeline({
  cards,
  streaming = false,
  toolCall = "",
  conversationId,
}: {
  cards: TurnCard[];
  streaming?: boolean;
  toolCall?: string;
  conversationId: string | null;
}) {
  if (streaming && cards.length === 0) {
    return (
      <div className="animate-rise w-full space-y-2.5">
        <PreparingCard toolCall={toolCall} />
      </div>
    );
  }

  return (
    <div className="animate-rise w-full space-y-2.5">
      {cards.map((card, index) => {
        if (card.type === "role") {
          return (
            <RoleCard
              key={index}
              phase={card}
              working={streaming && !card.done}
              toolCall={toolCall}
            />
          );
        }
        if (card.type === "handoff") {
          return (
            <HandoffCard
              key={index}
              variant={card.variant}
              text={card.text}
              conversationId={conversationId}
            />
          );
        }
        if (card.type === "failure") {
          return (
            <FailureCard
              key={index}
              text={card.text}
              code={card.code}
              conversationId={conversationId}
            />
          );
        }
        return (
          <TerminalResponseCard
            key={index}
            text={card.text}
            outcome={card.outcome}
            role={card.role}
            conversationId={conversationId}
          />
        );
      })}
    </div>
  );
}

/**
 * Fallback for assistant messages persisted before the `activity` timeline
 * existed: a neutral commentary card (if any) plus the answer card.
 */
export function LegacyAssistant({
  content,
  intermediateOutputs,
  conversationId,
  failure,
}: {
  content: string;
  intermediateOutputs?: IntermediateOutput[] | null;
  conversationId: string | null;
  /** Set on messages stored before the turn timeline, so they still read as failed. */
  failure?: TurnFailure | null;
}) {
  const notes = (intermediateOutputs || []).filter((output) => output && output.text);
  return (
    <div className="animate-rise w-full space-y-2.5">
      {notes.length ? (
        <Card
          avatar={<GearSix size={13} weight="fill" />}
          avatarClass="border-hair bg-panel2 text-zinc-400"
          title="Process"
          bodyLabel={`intermediate output · ${notes.length}`}
        >
          <NoteList
            notes={notes.map((output) => ({
              text: output.text,
              level: output.level === "milestone" ? "milestone" : "progress",
            }))}
            dotClass="bg-zinc-500"
          />
        </Card>
      ) : null}
      {failure ? (
        <FailureCard
          text={content}
          code={failure.code}
          conversationId={conversationId}
        />
      ) : (
        <TerminalResponseCard text={content} conversationId={conversationId} />
      )}
    </div>
  );
}
