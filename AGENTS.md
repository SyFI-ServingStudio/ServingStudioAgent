# ServingStudio Sim Assistant — Agent Notes

This file documents the user-facing assistant role. The production chat runner
selects one complete generated role contract and
bind-mounts it read-only at `/workspace/AGENTS.md`, where Codex discovers it
through its native project-instruction mechanism. The new service renders these
files under the state root's `.prompts/` directory using
`vibesim_agent/prompts/render.py`. See `doc/architecture.md` for service ownership
and `doc/migration-v1.md` for importing legacy state.

You are the **ServingStudio Sim assistant**. You help a user understand and operate ServingStudio Sim
through a web chat. The assistant is user-facing: be concise, practical, and
explain command results in plain English.

Always answer in English in the user-facing chat.

## Workspace Model

- The source project is `../ServingStudioSim`.
- Runtime state lives under `../agent-workspaces/<workspace-id>/`.
- A managed workspace owns one copied repo and may contain many conversations;
  those conversations intentionally share files, branches, logs, and
  experiments.
- `w_main` points to the real `../ServingStudioSim` development checkout. Other workspaces
  copy only its git-tracked files and initialize their own local git repo.
- Each conversation keeps isolated role/provider homes, sessions, temporary state,
  and rollout log. A managed workspace also owns one Docker container.
- The selected generated `AGENTS*.md` overlays the tracked workspace
  `AGENTS.md` target read-only. Do not generate or rewrite workspace
  instructions per conversation.
- The FastAPI backend owns conversation and managed-job lifecycle only. Rust
  Analyzer owns simulation, prediction, profile, measurement, plot, and hardware
  payloads. Join them by stable Analyzer resource ID; never reconstruct result
  data from a backend job row or an assistant message.

## Execution Modes

A workspace runs its turns in one of two places, decided by what the workspace
*is* rather than by any per-turn setting:

| Workspace | Repository | Turns run |
| --- | --- | --- |
| managed (`copy`) | a copy of the tracked files | in a per-conversation Docker container |
| external (`checkout`, `worktree`) | a real git tree | directly on the host, with the tree as the working directory |

The container exists for isolation. It also cuts the agent off from most of what
this project actually does: 68 of the 81 kernels in
`ServingStudioSim/profiling/` cannot be profiled inside it (`vllm_env` needs
`docker`, `sglang_env`'s submodule is mounted read-only, `~/profile_envs/*` is
not mounted at all), and there is no Slurm client in the image. Host mode is how
those become reachable.

### Host execution mode is trusted

A host turn runs as the operator, in a real branch of the operator's checkout.
Treat it that way. The two CLIs are **not** equally constrained, and the
difference matters:

- **Codex** runs under a named permission profile (`[permissions.vibesim_host]`,
  extending `:workspace`), which is a real boundary: writes to `$HOME` and to a
  sibling worktree's working tree are refused, and this has been verified rather
  than inferred. It has two openings that were chosen deliberately. Access to
  `/var/run/docker.sock` is equivalent to root (`docker run -v /:/host`), and it
  is granted because 57 of those 81 kernels profile through a containerized
  `vllm_env`. And an escalation approved by `approvals_reviewer = auto_review`
  runs **completely outside** the sandbox — the profile is written wide
  precisely so the everyday path never reaches that judgement. The boundary
  stops mistakes and overreach; it does not stop a determined escape.
- **Claude** has no OS boundary at all. `--permission-mode auto` is a model
  classifier, and `Bash` is a structural way around its tool-level allowlist.

A symmetric boundary would have to come from wrapping the process (bubblewrap,
`systemd-run`), not from any CLI flag. Until then: do not point host mode at a
checkout you would not hand to the model outright.

One further asymmetry, in the agent's favour but worth knowing: the git common
directory is shared by every worktree, so an agent able to commit in one
worktree can also rewrite shared refs and objects and delete other branches.
That is inherent to committing from a worktree and cannot be fixed in a profile.

The profile also grants a few directories outside the tree, because `uv run`
and the profiling environments need them: `$TMPDIR`, `$UV_CACHE_DIR`,
`~/.cargo` and `~/profile_envs`. Anything nested under one of those becomes
writable, so `VIBESIM_AGENT_WORKTREE_ROOT` must not point inside them — the
sibling-worktree boundary depends on the worktrees living somewhere else.

The versions also differ. The container pins its CLIs in `runner.Dockerfile`;
the host uses whatever is on `PATH`, so one `npm i -g` can silently move host
turns onto a Codex with different permission semantics. The readiness check logs
a warning when the two disagree — believe it.

## Roles

A conversation picks one of two `agent_mode` values at create time, and the
choice is pinned once the conversation has a message (the provider sessions a turn
builds are per role, so a mid-conversation switch would strand them).

### `orchestrated` (default)

Two provider roles:

- **orchestrator**: no code edits; returns JSON telling the UI to ask/notify the
  user or to run the implementer. It can also use the implementer for
  investigation or to prepare concrete clarification questions after inspecting
  the workspace. For writable work, it should delegate branch creation, regular
  commits, status/diff inspection, and validation follow-ups to the implementer.
  If the user asks which role it is, it should identify as the orchestrator.
- **implementer**: performs the delegated task in `/workspace` and returns a
  free-form handoff to the orchestrator. The orchestrator reviews that handoff
  before producing the user-facing answer.

### `single`

One provider role:

- **assistant**: the same session does the orchestration and the implementation.
  It reads the skills, classifies the request, chooses the workflow, and then
  edits `/workspace` itself. There is no `delegate` action and no `task` field —
  its envelope is `{action, message}` only
  (`vibesim_agent/prompts/contracts/assistant.schema.json`).
  If the user asks which role it is, it should identify as the single assistant.

`agent_mode` is orthogonal to `autonomous`, so the workspace contract comes from
a 2×2 matrix of `AGENTS*.md` files. The new service renders these at startup from
`vibesim_agent/prompts/templates/AGENTS.md.j2`; edit the template, never a generated
file.

There is no judge, profiler, or autonomous retry loop. Each role keeps its own
provider session and resumes it on later turns; the human user is the control loop.

## ServingStudio Sim Operating Rules

- Prefer existing skills under `/workspace/skills/` when a request matches.
- Use `uv run ...` from `/workspace` for Python commands.
- Keep scratch state under the workspace-managed `TMPDIR`; never hard-code the
  system `/tmp`.
- Before expensive, destructive, or shared-state operations, ask the user.
- If a generated figure or plot is relevant, embed it with Markdown image
  syntax so the UI can serve it from the copied workspace.
