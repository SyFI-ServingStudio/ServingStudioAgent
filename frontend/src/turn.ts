import type { CodexRoleRuntime, CommentaryLevel, Role, RoleNote, RolePhase, TurnCard, TurnEvent } from "./types";

const ROLES: readonly Role[] = ["orchestrator", "implementer", "assistant"];

function asRole(value: string): Role {
  return (ROLES as readonly string[]).includes(value) ? (value as Role) : "orchestrator";
}

/**
 * Clean an orchestrator commentary note. The orchestrator sometimes narrates by
 * emitting its whole decision envelope (`{"action":…,"message":…,"task":…}`)
 * into the commentary channel; show the human `message`, drop a bare decision
 * envelope (surfaced as its own card), otherwise leave the text as-is. Mirrors
 * the backend `unwrap_commentary` so already-persisted notes reload cleanly too.
 */
export function cleanNote(text: string, level?: CommentaryLevel): RoleNote | null {
  const stripped = text.trim();
  if (!(stripped.startsWith("{") && stripped.endsWith("}"))) {
    return { text, level: level ?? "progress" };
  }
  try {
    const payload = JSON.parse(stripped) as Record<string, unknown>;
    if (payload && typeof payload === "object") {
      const message = payload.message;
      if (typeof message === "string" && message.trim()) {
        return {
          text: message.trim(),
          level: payload.action === "milestone" ? "milestone" : (level ?? "progress"),
        };
      }
      if ("action" in payload || "task" in payload) {
        return null;
      }
    }
  } catch {
    return { text, level: level ?? "progress" };
  }
  return { text, level: level ?? "progress" };
}

/**
 * Fold an ordered turn-event stream into the role-timeline cards.
 *
 * Segmentation (events arrive in this order from the backend):
 *  - `intermediate_output` accumulates notes into the current role phase; a role
 *    switch or a closed phase opens a new one (round++ per role).
 *  - `usage` closes the current role phase and attaches its duration + tokens.
 *  - `decision` emits the orchestrator→implementer delegated-task hand-off card.
 *  - `implementer` emits the implementer→orchestrator conclusion hand-off card.
 *  - `final` emits the answer card.
 *  - `error` emits the failure card, the terminal entry the backend persists in
 *    place of `final` whenever a turn ends without an answer.
 *
 * The same reducer runs live (incremental events) and on reload (persisted
 * `activity`), so a reopened conversation rebuilds the identical timeline.
 */
export function reduceTurn(events: TurnEvent[]): TurnCard[] {
  const cards: TurnCard[] = [];
  const round: Record<Role, number> = { orchestrator: 0, implementer: 0, assistant: 0 };
  let current: RolePhase | null = null;
  // The terminal envelope arrives without a role of its own, so the response
  // card inherits whichever role last spoke. In single mode that is `assistant`,
  // and the "Input needed" card must not be tinted as the orchestrator.
  let lastRole: Role = "orchestrator";

  const runtimeFrom = (event: { model?: string; effort?: string }): CodexRoleRuntime | undefined =>
    event.model
      ? { model: event.model, effort: event.effort ?? "", serviceTier: "default" }
      : undefined;

  const openPhase = (role: Role, runtime?: CodexRoleRuntime): RolePhase => {
    round[role] += 1;
    const phase: RolePhase = {
      type: "role",
      role,
      runtime,
      round: round[role],
      notes: [],
      durationMs: null,
      tokens: null,
      done: false,
    };
    cards.push(phase);
    current = phase;
    return phase;
  };

  const lastOpenPhase = (role: Role): RolePhase | null => {
    for (let i = cards.length - 1; i >= 0; i -= 1) {
      const card = cards[i];
      if (card.type === "role" && card.role === role && !card.done) {
        return card;
      }
    }
    return null;
  };

  for (const event of events) {
    switch (event.kind) {
      case "intermediate_output": {
        const clean = cleanNote(event.text, event.level);
        if (!clean) {
          break;
        }
        const role = asRole(event.role);
        const phase: RolePhase =
          current && current.role === role && !current.done
            ? current
            : openPhase(role, runtimeFrom(event));
        phase.runtime = runtimeFrom(event) ?? phase.runtime;
        phase.notes.push(clean);
        current = phase;
        lastRole = role;
        break;
      }
      case "usage": {
        const role = asRole(event.role);
        const target: RolePhase =
          (current && current.role === role && !current.done ? current : null) ??
          lastOpenPhase(role) ??
          openPhase(role, runtimeFrom(event));
        target.runtime = runtimeFrom(event) ?? target.runtime;
        target.durationMs = event.duration_ms;
        target.tokens = event.tokens;
        target.done = true;
        lastRole = role;
        if (current === target) {
          current = null;
        }
        break;
      }
      case "decision": {
        cards.push({ type: "handoff", variant: "delegated_task", text: event.task });
        current = null;
        break;
      }
      case "implementer": {
        cards.push({ type: "handoff", variant: "conclusion", text: event.text });
        current = null;
        break;
      }
      case "error": {
        cards.push({ type: "failure", text: event.text, code: event.code });
        current = null;
        break;
      }
      case "final": {
        cards.push({
          type: "response",
          text: event.text,
          outcome: event.outcome ?? "final_answer",
          role: lastRole,
        });
        current = null;
        break;
      }
    }
  }

  return cards;
}

/** "12s" / "1m 3s" / "<1s" from a millisecond duration. */
export function formatDuration(ms: number | null): string {
  if (ms === null || ms <= 0) {
    return "";
  }
  const seconds = Math.round(ms / 1000);
  if (seconds < 1) {
    return "<1s";
  }
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
}

/** "12.3k" / "930" compact token count. */
export function formatTokens(value: number): string {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}k`;
  }
  return String(value);
}

/** Strip the leading role tag from one transient tool-call line. */
export function stripRolePrefix(text: string): string {
  return text.replace(/^(orchestrator|implementer|assistant):\s*/i, "");
}
