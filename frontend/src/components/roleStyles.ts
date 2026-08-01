import { Brain, Wrench, type Icon } from "@phosphor-icons/react";

import type { Role } from "@/types";

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
}

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
  },
};
