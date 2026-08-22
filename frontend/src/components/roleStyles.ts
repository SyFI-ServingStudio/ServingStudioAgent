import { Brain, Sparkle, Wrench, type Icon } from "@phosphor-icons/react";

import type { AgentMode, Role } from "@/types";

/**
 * Per-role class bundles. Tailwind only emits classes it sees as *literal*
 * strings in source, so every role-tinted class lives here as a full literal —
 * never `border-${color}-line` at runtime.
 */
export interface RoleStyle {
  label: string;
  Icon: Icon;
  /** icon avatar tile */
  avatar: string;
  /** note bullet + tool-call activity dots background */
  dot: string;
  /** "working" status chip */
  chipWorking: string;
  /** working-card border + ring */
  cardWorking: string;
  /** tool-call activity line text color */
  progressText: string;
  /** quiet left-accent bar keyed to the role (2px, low opacity) */
  accent: string;
  /** faint flow-tinted card background */
  tint: string;
  /** role name as inline label text (runtime picker, chips) */
  labelText: string;
}

/**
 * Which roles a conversation drives, by agent mode. Mirrors the backend's
 * `AGENT_MODE_ROLES`; the UI must not show a picker for a role that mode never
 * spawns, or its model choice would read as active when it is inert.
 */
export const AGENT_MODE_ROLES: Record<AgentMode, readonly Role[]> = {
  orchestrated: ["orchestrator", "implementer"],
  single: ["assistant"],
};

export const ROLE_STYLE: Record<Role, RoleStyle> = {
  orchestrator: {
    label: "Orchestrator",
    Icon: Brain,
    avatar: "border-orch-line bg-orch-bg text-orch",
    dot: "bg-orch",
    chipWorking: "border-orch-line bg-orch-bg text-orch-soft",
    cardWorking: "border-orch-line ring-1 ring-orch/15",
    progressText: "text-orch",
    accent: "border-l-2 border-l-orch/45",
    tint: "tint-orch",
    labelText: "text-orch-soft",
  },
  implementer: {
    label: "Implementer",
    Icon: Wrench,
    avatar: "border-impl-line bg-impl-bg text-impl",
    dot: "bg-impl",
    chipWorking: "border-impl-line bg-impl-bg text-impl-soft",
    cardWorking: "border-impl-line ring-1 ring-impl/15",
    progressText: "text-impl",
    accent: "border-l-2 border-l-impl/45",
    tint: "tint-impl",
    labelText: "text-impl-soft",
  },
  // Single-agent mode. It plays both parts, so it gets its own hue rather than
  // reusing the orchestrator's — a turn is never a mix of the two.
  assistant: {
    label: "Assistant",
    Icon: Sparkle,
    avatar: "border-asst-line bg-asst-bg text-asst",
    dot: "bg-asst",
    chipWorking: "border-asst-line bg-asst-bg text-asst-soft",
    cardWorking: "border-asst-line ring-1 ring-asst/15",
    progressText: "text-asst",
    accent: "border-l-2 border-l-asst/45",
    tint: "tint-asst",
    labelText: "text-asst-soft",
  },
};
