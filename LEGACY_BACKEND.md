# Legacy Backend Deployment Reference

This preserves the pre-cutover README for the retained `e26ad6d` deployment.
Its source was removed from the new service checkout. Commands and environment
keys below require that separate old checkout unless labeled as refactor entries.
For the new service, start with [README.md](README.md).

# VibeSim Agent Backend

A shared workspace/conversation backend for VibeSim. The integrated VibeSimUI
talks to this FastAPI service, which drives **Codex or Claude Code inside Docker** and
records durable workspace, conversation, turn, and managed-job ownership state.
Rust Analyzer remains the read-only authority for result catalogs and payloads.

For a complete deployment, clone the
[`VibeSimWorkspace`](https://github.com/serendipity-zk/VibeSimWorkspace)
meta-repository and follow its `reproduce.md`.

A workspace owns one repo/logs root and may contain many conversations. A
managed workspace gets one isolated copy of git-tracked files from `../VibeSim`;
all conversations in that workspace reuse it. `w_main` points at the real
development checkout and is never copied or rewritten by workspace creation.
Each conversation still gets its own Codex home and Docker container.

## Refactor Entry

The new `vibesim_agent` package has explicit commands for isolated validation.
The existing `run.sh` still starts the baseline backend until migration and
deployment cutover are validated.

```bash
uv run python -m vibesim_agent env-reference
```

For fresh state, set `VIBESIM_AGENT_MAIN_DIR` to the absolute VibeSim checkout
path and `VIBESIM_AGENT_WORKSPACES_ROOT` to a new absolute directory outside that
checkout. Initialization creates the external `w_main` descriptor, current
SQLite schema and registry index; it does not copy or modify the main checkout
or its logs. Any existing state directory is rejected, including an empty one.

```bash
uv run python -m vibesim_agent init
```

Before serving, select and check the bind address and port using the workspace
`reproduce.md` per-user convention. Set `VIBESIM_AGENT_BIND`,
`VIBESIM_AGENT_PORT`, `VIBESIM_AGENT_MANAGED_BACKEND_URL` and
`VIBESIM_AGENT_ANALYZER_BASE_URL` for that deployment; callback and Analyzer URLs
must be reachable from the runner. Runner images must already be built.

```bash
uv run python -m vibesim_agent serve
```

The service uses one process. Its factory, `vibesim_agent.bootstrap:create_application`,
checks all workspace databases and acquires the state directory lock before
writing generated prompts. Lifespan startup recovers interrupted turns before
accepting requests; shutdown drains activity before releasing ownership.
Existing databases with an older format require managed startup or offline
migration, never `init`.
Retired Agent environment keys are rejected by name, without printing their values.
Provider defaults use `VIBESIM_PROVIDER_<ID>_*`; role defaults remain GPT, with
per-conversation model selection available through the API.

For a reviewed legacy deployment, `serve --startup-config /absolute/startup.json`
detects the stored format, stops the explicitly audited legacy stack when needed,
migrates into an independent target, and persists its selection for later starts.
`selected-root --startup-config /absolute/startup.json` reads that same selection
for Analyzer without starting a migration. Both commands require the same provider
configuration. See [managed startup configuration](doc/migration-v1.md#managed-startup-entry)
for the deployment file and shutdown constraints. The existing production scripts
have not been switched to this entry.

## Runner Image Preparation

The refactor has an explicit image build entry point. Use a separate image tag
for rehearsal and pass that same tag to the rehearsal backend:

```bash
VIBESIM_RUNNER_IMAGE="vibesim-agent-runner:${USER}-refactor" \
  ./scripts/build-runner-image.sh
```

It uses the same `VIBESIM_AGENT_MAIN_DIR` and `VIBESIM_RUNNER_*` runtime
configuration as the new backend, including image tag, UID/GID and user. The
source must be a Git repository root with `uv.lock`. It copies tracked working
files and required initialized submodules into a private build context; newly
added source files must be tracked before building. It does not include
untracked workspace state or supply provider credentials as build arguments.

Additional build-only overrides are `VIBESIM_RUNNER_CUDA_IMAGE`,
`VIBESIM_RUNNER_UV_IMAGE`, `VIBESIM_RUNNER_CODEX_NPM_PACKAGE`,
`VIBESIM_RUNNER_CLAUDE_NPM_PACKAGE`, `VIBESIM_RUNNER_NODE_VERSION`,
`VIBESIM_RUNNER_NODE_ARCH`, and `VIBESIM_RUNNER_RUST_TOOLCHAIN`.
`VIBESIM_RUNNER_SKIP_IMAGE_TEST=1` explicitly skips post-build acceptance;
the default is `0`, which runs the build smoke test against the same source
directory. A build or smoke failure exits unsuccessfully.

The current Dockerfile provides `/home/<runner-user>`, `/opt/vibesim-venv`,
`/opt/vibesim-uv-cache`, and `DG_USE_LOCAL_VERSION=0`. The new entry rejects
incompatible runtime overrides before building; other images may support them.
The new entry uses `docker/runner.Dockerfile` and `scripts/test-runner-image.sh`,
with the default image `vibesim-agent-runner:<runner-user>` and runner version
`prebuilt-agent-runner-v12`. Image labels use `org.vibesim.agent.*`.
Node.js is pinned to `v22.23.2` to satisfy the pinned Claude CLI's Node >=22
requirement; npm rejects an incompatible engine during Claude installation.
The legacy Dockerfile and scripts remain available until deployment cutover.
The smoke script reads the same new configuration; `timing` mode requires a
nonempty `VIBESIM_RUNNER_GPUS`. It reads the GPU model inside the container and
runs a private copy of the timing preset with that model and fresh output paths.
Mixed visible GPU models are rejected; select same-model devices with
`VIBESIM_RUNNER_GPUS`. Profiling retains the launcher's idle-device checks.
Failed smoke runs that produced logs retain their private workspace and print
its path for diagnosis; successful runs remove the private copy.
No image is built at Agent
service startup. The image retains dependency caches; first-party simulator
and Analyzer binaries still build in the workspace when the launcher needs them.
The image does not currently provide Docker for nested profiling containers.
A cold profile cache that needs a container-backed kernel, such as the preset's
`kv_cache_append:vllm_cuda`, fails with `docker is required for container profiling`.
GPU visibility and prebuilt dependency caches alone do not satisfy that runtime
requirement; full cold-cache timing acceptance is still outstanding.

## Workspace model

The refactored server's unified managed-job callbacks and legacy compatibility
are documented in [Managed Job Callbacks](doc/managed-jobs.md).

Runtime state lives under `../agent-workspaces/`:

```text
agent-workspaces/
├── registry.json
├── w_main/
│   ├── workspace.json
│   └── workspace.sqlite
└── w_<id>/
    ├── workspace.json
    ├── workspace.sqlite
    ├── repo/
    ├── codex/<conversation-id>/
    └── jobs/
```

`workspace.json` is the human-readable identity/location descriptor.
Conversations, messages, sessions, turns, jobs, experiments and
conversation-to-experiment relationships live in that workspace's SQLite.
`registry.json` is the bounded discovery surface consumed by the read-only
Analyzer. Archiving a workspace removes its logs from active discovery;
removing a conversation container does not remove the workspace repo or logs.

The normal browser API is workspace-scoped:

```text
GET/POST /api/agent/v1/workspaces
GET/PATCH /api/agent/v1/workspaces/{workspace_id}
GET/POST /api/agent/v1/workspaces/{workspace_id}/conversations
GET/DELETE /api/agent/v1/workspaces/{workspace_id}/conversations/{conversation_id}
POST /api/agent/v1/workspaces/{workspace_id}/conversations/{conversation_id}/messages
GET  /api/agent/v1/workspaces/{workspace_id}/conversations/{conversation_id}/stream
POST /api/agent/v1/workspaces/{workspace_id}/conversations/{conversation_id}/cancel
GET  /api/agent/v1/workspaces/{workspace_id}/conversations/{conversation_id}/experiments
```

The backend root still serves a compatibility/debug chat shell; VibeSimUI is the
primary browser entry. Its read-only `GET /api/agent/v1/conversations` index flattens
active workspace summaries and
includes `workspace_id`; the shell immediately converts each row to the
workspace-scoped routes above for load, pagination, send, reconnect, cancel,
delete, and local images. This compatibility index preserves access to migrated
history without reintroducing conversation-owned workspaces or unscoped writes.

Opening a workspace or Analyzer panel only lists/restores history. An empty
conversation is materialized on the first actual send, not on page mount.

Workspaces created by the browser entry page and their new conversations begin
with `naming_state: "pending"`. After the first successful answer, a best-effort
OpenRouter request generates one stable workspace name and one conversation
title. The answer is returned before naming runs. A manual workspace rename
sets `naming_state: "manual"` and permanently wins over delayed generation.
Existing/migrated objects default to `manual`; no history is backfilled.

Legacy `conversations.json` plus `workspaces/<conversation-id>/` data can be
audited and migrated with:

```bash
UV_CACHE_DIR="$TMPDIR/uv-cache-user-facing-ui" \
uv run python -m backend.migrate_workspaces --dry-run

# Execute only after reviewing the exact plan and storage impact:
UV_CACHE_DIR="$TMPDIR/uv-cache-user-facing-ui" \
uv run python -m backend.migrate_workspaces --execute
```

Execution uses same-filesystem atomic renames, validates source/destination tree
hashes and imported message counts, then archives only the now-empty legacy
container plus JSON. It does not duplicate the legacy repo trees before moving
them.

Migration builds that predate exact timestamp restoration can be repaired from
their immutable archive. The command refuses to touch a conversation whose
messages changed after migration:

```bash
UV_CACHE_DIR="$TMPDIR/uv-cache-user-facing-ui" \
uv run python -m backend.migrate_workspaces \
  --repair-completed ../agent-workspaces/migrations/<timestamp>
```

## Run

### Claude backend

Each active role can select Claude independently: use Claude in `single` mode,
or combine a Codex orchestrator with a Claude implementer. The existing model
picker discovers the Claude family through `/api/agent/v1/codex-backends`; no separate
browser build is required to populate it. The `codex_runtime` API field and
`codex_sessions` table retain their historical names for compatibility.

Provide one credential in the **backend process environment**:

- `ANTHROPIC_API_KEY` for the Anthropic API.
- `ANTHROPIC_AUTH_TOKEN` for a compatible gateway, with `ANTHROPIC_BASE_URL`.
- `CLAUDE_CODE_OAUTH_TOKEN` for a token provisioned for Claude Code automation.

Only credential presence is checked in the catalog; account access and model
availability are checked by the actual call. No host Claude login/settings or
conversation history is copied. Credentials are passed to Docker by environment
variable name, never embedded in CLI arguments. Keep only the intended auth
method configured. Calls receive current backend credentials even when reusing
a container.

The catalog offers explicit `claude-sonnet-5` and `claude-opus-5` model IDs,
with `low`/`medium`/`high`/`xhigh`/`max` efforts and the `default` service tier.
Both models passed live structured-output requests at `xhigh` and `max` through
this deployment's configured gateway. Its `/v1/models` endpoint returns 403,
so this is a verified selection, not an exhaustive provider model inventory.
Existing `sonnet`/`opus` selections resolve to these pinned versions. Set
`CLAUDE_MODEL` to add a custom model ID (unknown models retain conservative
low/medium/high efforts).

Rebuild the runner with `./scripts/build-codex-runner-image.sh` before enabling
Claude. The image installs `@anthropic-ai/claude-code@2.1.250`; the build argument
`CLAUDE_NPM_PACKAGE` can override that pin. The runner label is now
`prebuilt-agent-runner-v11`, which also makes `run.sh` rebuild an older image.
For an isolated development deployment, set a distinct `CODEX_DOCKER_IMAGE`,
`VIBESIM_WORKSPACES_ROOT`, and `PORT` before building or starting the backend.

Each role persists Claude state under `codex/<conversation-id>/<role>/claude/`.
The same role resumes its saved Claude session on later turns. Switching to a
different family after messages exist remains disallowed. Instructions are
explicitly loaded from the selected `/workspace/AGENTS.md`; the role's skills
directory points at `/workspace/skills`. Analyzer MCP is configured explicitly.
Tool permissions follow the same Docker/role-instruction policy as the Codex
runner. Only the final validated `structured_output` can route a turn; streamed
progress and tool events cannot trigger delegation or finish a conversation.

Run the CPU regression suite with:

```bash
uv run python -m unittest discover -s tests
```

It covers CLI protocol conversion, real subprocess pipes, timeout/cancellation,
single and mixed-role dispatch, persistence, and the existing Codex behavior.
It makes no model requests. After providing credentials and rebuilding the
image, validate a new single-Claude conversation with a small file-read task,
resume it with a follow-up, cancel an active task, and repeat with a Codex
orchestrator plus Claude implementer. These live checks require model access
and the running Analyzer service for MCP queries.

Claude protocol references: [programmatic usage](https://code.claude.com/docs/en/headless)
and [CLI reference](https://code.claude.com/docs/en/cli-reference).

### Start the service

```bash
cd VibeSimAgent
./run.sh
# then open http://<host>:8765
```

`run.sh` first builds the React/Vite frontend into `frontend/dist`, then starts
the FastAPI backend. If `frontend/node_modules` is missing, it runs `npm ci`
once before the build. Set `FRONTEND_SKIP_BUILD=1` when you are already running
the Vite dev server.

The browser-facing `GET /api/agent/v1/jobs` endpoint is a read-only ownership/lifecycle
overlay for non-simulation managed results across active workspaces. Each row
carries workspace/conversation/job identity, lifecycle status, a stable backend
`resource_id`, and the corresponding `analyzer_resource_id`. It intentionally
does **not** expose artifact paths, descriptors, summaries, curves, or plots.
Rust Analyzer owns those result catalogs and payloads; VibeSimUI joins the two
sources by `analyzer_resource_id`. Simulation sweeps are discovered directly
from Analyzer and linked to conversations through experiment relationships.

The backend uses a prebuilt local Docker image for the Codex runner. If the image
is missing, its VibeSim runner label is stale, or its baked `VibeSim/uv.lock` hash
does not match the current checkout, `run.sh` builds it once from
`docker/codex-runner.Dockerfile`; later turns and later conversations reuse that
image. The image is based on CUDA 12.8 devel and includes Node/Codex, `uv`, git,
Rust stable (`cargo`/`rustc`), `just`, `nvcc`, Python 3.12 dev headers, native
build tools, and a prewarmed VibeSim Python environment at `/opt/vibesim-venv`.
The image build copies the tracked `main` tree, prewarms the default profiling
stack (including pinned DeepGEMM), and compiles the full Cargo release target for
the simulator and analyzer once to populate third-party dependency artifacts.
Before the target is stored at `/opt/vibesim-cache/target`, the image removes all
first-party VibeSim crate artifacts and final binaries. A new conversation copies
this dependency-only seed into its empty `/workspace/target`; Cargo then compiles
that workspace's exact VibeSim revision while reusing heavyweight dependencies
such as DataFusion and Arrow. The runner image is therefore coupled to its
toolchain version and Python `uv.lock`, not to a mutable source SHA.
The Docker container and `codex exec` both run as the host UID/GID with `HOME`
set to the matching `/home/<user>` path.

To rebuild the runner image explicitly:

```bash
cd VibeSimAgent
export CODEX_DOCKER_IMAGE="vibesim-ui-codex-runner:${USER}"
CODEX_FORCE_IMAGE_BUILD=1 ./scripts/build-codex-runner-image.sh
```

Use a user-specific tag on hosts with a shared Docker daemon. Runner images
embed the building user's UID/GID and home path, so a shared `latest` tag can be
valid for one account and unusable by another. Keep the same
`CODEX_DOCKER_IMAGE` value when starting `run.sh`.

### Runner image acceptance test

After building the image, run the non-GPU acceptance test:

```bash
cd VibeSimAgent
./scripts/test-codex-runner-image.sh build
```

This creates a temporary copy of the tracked `main` tree and runs the image as
the same non-root UID/GID used in production. It requires writable Cargo and uv
caches, verifies the repository-selected mold linker and `protoc`, seeds the
workspace from the image's complete release `target`, verifies Cargo accepts the
seed for the simulator and analyzer, and runs the launcher's read-only cache
report without analyzer output.
The temporary workspace and container are removed when the test exits; the real
`main` checkout and its `profile.db` are never mounted into the container.

On a GPU host with a warm tracked kernel catalog, the stronger mode also runs a
real Llama timing prediction and requires its analyzer artifact:

```bash
./scripts/test-codex-runner-image.sh timing
```

The GPU mode is explicit so ordinary image builds remain hardware-independent.
The image builder runs the `build` gate after `docker build`; set
`CODEX_SKIP_RUNNER_IMAGE_TEST=1` only for an explicitly incomplete development
build that must not be treated as ready for agent conversations.

The compatibility chat shell can run its Vite frontend against the same backend:

```bash
cd VibeSimAgent
./run.sh
# in another shell
cd frontend
npm run dev
# then open http://<host>:5173
```

When the backend is intentionally bound only to a Docker bridge address, expose
the same process on loopback without starting a second backend:

```bash
uv run python scripts/localhost_forward.py \
  --upstream-host 172.19.0.1 --upstream-port 8765
# then open http://127.0.0.1:8765
```

Run the forwarder under the host's process supervisor or a detached tmux
session when the loopback endpoint must outlive the current shell.

## Turn Flow

```text
browser
  -> FastAPI
  -> select agent-workspaces/<workspace-id>/repo (or the external w_main checkout)
  -> seed agent-workspaces/<workspace-id>/codex/<conversation-id> from host ~/.codex auth/config
  -> docker run -v <workspace-repo>:/workspace -v <conversation-codex-home>:/home/<user>/.codex
     - overlay the selected backend/prompts/AGENTS*.md read-only at /workspace/AGENTS.md
     - repo-local skills remain in /workspace/skills
     - when host HF_HOME is set, mount it read-only at /model and set container HF_HOME=/model
     using the prebuilt CODEX_DOCKER_IMAGE
  -> codex exec/resume as the driving role
       (agent_mode=orchestrated -> orchestrator; agent_mode=single -> assistant)
       action=progress           -> emit a concise user-facing update and continue
       action=milestone          -> emit a completed checkpoint and continue
       action=final_answer       -> return completed results to the user
       action=request_user_input -> return a blocking question to the user
       action=delegate           -> codex exec/resume as implementer
                                    (orchestrated only; in single mode the
                                     envelope has no delegate action, and one
                                     emitted anyway is sent back for repair)
       action=reply_user         -> (implementer only, and only when the prompt
                                     carried the user's own words) answer the
                                     user and end the turn without the driver
  -> explicit handoff of implementer summary back to orchestrator  [orchestrated only]
       action=final_answer       -> return reviewed result to the user
       action=request_user_input -> request genuinely required user input
       action=delegate           -> continue with another bounded implementer task
  -> retain the active turn independently of the browser connection
       GET .../stream -> replay and continue after refresh
       GET .../stream -> 204 when the conversation is idle
       POST .../cancel -> send SIGINT to Codex and stop the whole turn
  -> stream tool-call activity, semantic commentary, and one terminal response
       outcome=final_answer       -> render the completed Answer state
       outcome=request_user_input -> render Input needed and focus the composer
     Historical activity without an outcome remains compatible and defaults to
     final_answer.
  -> when Launcher sees the managed capability context:
       simulations register through the compatible managed-runs protocol
       timing-predict and kernel profiling register typed managed jobs
       approve a bounded logs-root artifact path before artifacts are created
       stream requested/running/analysis/ready lifecycle as durable job events
       preserve simulation experiment identity and typed result-resource identity
```

Typed jobs currently include `timing_predict`, `kernel_profile`, and
`kernel_measure`. They are linked directly to the conversation without being
misclassified as deployment simulations. A ready typed-job card resolves its
stable Analyzer ID and opens the corresponding Analyzer-owned prediction,
profile, or measurement surface. The conversation backend never reads
`curve.json`, summary JSON, or plot files to construct UI results. The mutable
`profile.db` remains the L1 cache authority; immutable per-invocation Analyzer
artifacts are the displayed-result authority.

The orchestrator is normally an active human-in-the-loop coordinator. It reads
matching skills, classifies the request, decides whether clarification is
needed, and delegates only bounded implementer tasks. Before the first message
in a conversation, the welcome area shows an Autonomous button below the example
questions. When enabled, the backend mounts
`backend/prompts/AGENTS.autonomous.md` at `/workspace/AGENTS.md` instead; that
contract tells the orchestrator to proceed with conservative assumptions
instead of asking preference or clarification questions. After the first user
message, the conversation's autonomous setting is fixed.

Beside it sits a Single agent / Orchestrated button. `agent_mode=single`
replaces the two-role loop with one `assistant` role that both plans and
implements: it costs one Codex call per user message rather than `1 + 2D`, keeps
one Codex session instead of two, and needs no handoff text because nothing is
handed off. Its envelope drops `delegate` and `task`
(`backend/prompts/assistant.schema.json`); a `delegate` decision emitted anyway
is rejected at parse time and repaired through the normal repair budget. Like
autonomous, the mode is fixed after the first user message — the Codex sessions
a turn builds are per role, so switching would strand them and restart the new
role with no history. The two switches are orthogonal, which is why there are
four `AGENTS*.md` variants.

### Prompt rendering

The mode-dependent prompts are **generated build output and are gitignored**:
the four `AGENTS*.md` variants come from
`backend/prompt_templates/AGENTS.md.j2`, and `orchestrator.txt` /
`assistant.txt` from `backend/prompt_templates/role.txt.j2`. Only the templates
are source. The rest of `backend/prompts/` — the two `*.schema.json`,
`implementer.txt`, `naming-*.txt` — has no mode variants, is hand-written, and
stays tracked.

They exist as files because the runtime consumes them as files (bind-mount
source, container-reuse `cmp -s`, fingerprint hashing), so
`backend/codex_runtime/config.py` calls `agents_prompt.ensure_rendered()` when
it is imported. That covers every entry point — `./run.sh`, a bare `uvicorn`,
`unittest discover` — including a fresh clone where the files do not yet exist.
Jinja2 is therefore a runtime dependency, not a dev-only one.

To change a contract, edit the template and read what it produced:

```bash
$EDITOR backend/prompt_templates/AGENTS.md.j2
uv run python -m backend.agents_prompt   # re-render now instead of on next import
git diff --no-index /dev/null backend/prompts/AGENTS.autonomous.md | less
```

Reviewing the rendered text matters more than usual here: a Jinja whitespace or
condition mistake produces a plausible contract rather than an error, and it can
land in only one of the four variants. Nothing enforces re-rendering, because
nothing has to — a hand-edited artifact is silently overwritten on the next
import, and `tests/test_prompts.py` asserts exactly that.

The implementer returns its own two-action envelope — `final_answer` back to the
orchestrator, or `reply_user` straight to a user who interrupted it and asked it
something. There is no judge, profiler, or shared
`profile.db` write unless the copied workspace task does it. The orchestrator
and implementer keep separate Codex session ids. Same-role continuity uses
`codex exec resume`; cross-role handoff does not rely on shared context and is
passed explicitly as task text and implementer summary.
The four `backend/prompts/AGENTS*.md` files hold the
detailed shared role and skill instructions. One is mounted read-only into each
conversation container as `/workspace/AGENTS.md`; this preserves Codex's native
project-instruction discovery and per-conversation mode without changing shared
workspace contents. The tracked blank `VibeSim/AGENTS.md` is a fail-safe bind
target—Codex skips it outside the managed container, and Docker refuses to start
if a workspace lacks that target. The workspace keeps `.codex/skills ->
../skills` so Codex can discover the copied repo-local skills. Later turns
resume that role and send the new user message or delegated task. Every role
call, including resumed Orchestrator calls and Implementer→Orchestrator
handoffs, prefixes the payload with the current role contract. Prompt
fingerprints remain provenance only: editing a prompt never discards the
durable Codex session or its history. The Orchestrator also receives the stable
conversation id and maintains `<conversation-id>_plan.md` plus
`<conversation-id>_progress.md` as concise recovery state in the workspace.
Implementer summaries are explicitly sent back to the orchestrator before the
turn finishes.

The UI shows assistant intermediate output and, when work is delegated, the
implementer summary. If the orchestrator delegates multiple follow-ups in one
browser turn, the final assistant message includes the implementer summaries and
the orchestrator's final user-facing message.

Runtime startup failures are persisted and emitted as a bounded
`failure: {code,message}` object. The user-facing message is safe and retryable;
the complete subprocess or Docker diagnostic remains in the structured backend
log instead of being rendered as an assistant answer.

Long browser conversations load backwards in fixed-size message pages. The
browser requests the newest page with
`GET /api/agent/v1/workspaces/{wid}/conversations/{cid}?limit=<n>` and requests an older page with
`?limit=<n>&before=<start_index>`, where `start_index` comes from the current
response's `message_page`. Reaching the top of the message viewport triggers the
older request and preserves the visible scroll position while prepending it.
Calling the same endpoint without query parameters retains the original
full-history response. Pagination never trims stored messages or changes role
session continuity.

## Agent API

Besides the browser UI, VibeSim exposes a small **HTTP surface for other agents**
to call. It is self-describing: fetch `GET /api/agent/v1/tools/skill` to get the full skill
(`SKILL.md` — what VibeSim does, when to call it, what to expect, and the
contract), then drive everything with plain HTTP — no framework glue.

The **real interactive interface** is the agent conversation API: multi-turn,
synchronous JSON, with workspace + Codex-session continuity across turns (it
reuses the same `store` and `run_turn` as the browser SSE path). The calling
agent reads each turn's `final` and, like a human, answers clarifying questions
or steers with another turn. `/api/agent/v1/tools/eval` is **evaluation-only** (single-turn,
stateless; for testcases).

| Method   | Path                                                       | Auth   | Purpose                                                         |
| -------- | ---------------------------------------------------------- | ------ | --------------------------------------------------------------- |
| GET      | `/api/agent/v1/tools/skill`                                         | public | Agent skill doc (`SKILL.md`, `text/markdown`).                  |
| GET/POST | `/api/agent/v1/tools/workspaces`                                    | token  | List or create durable workspaces.                              |
| POST     | `/api/agent/v1/tools/workspaces/{wid}/conversations`                | token  | Create an interactive conversation.                             |
| POST     | `/api/agent/v1/tools/workspaces/{wid}/conversations/{cid}/messages` | token  | Run one turn; synchronous JSON.                                 |
| GET      | `/api/agent/v1/tools/workspaces/{wid}/conversations/{cid}`          | token  | Full conversation history.                                      |
| DELETE   | `/api/agent/v1/tools/workspaces/{wid}/conversations/{cid}`          | token  | Human/operator cleanup only; calling agents must not invoke it. |
| GET      | `/api/agent/v1/tools/workspaces/{wid}/artifacts`                    | token  | List workspace files.                                           |
| GET      | `/api/agent/v1/tools/workspaces/{wid}/artifacts/download`           | token  | Download one workspace file.                                    |
| POST     | `/api/agent/v1/tools/eval`                                                | token  | Single-turn evaluation only (not interactive).                  |

**Auth** is gated by `VIBESIM_API_TOKEN`. When it is set, agent endpoints require
`Authorization: Bearer <token>` (missing/wrong → `401`); `/api/agent/v1/tools/skill` stays
public. When it is unset (local dev / same-host eval harness), no header is
needed. The browser UI routes (`/api/agent/v1/workspaces*`, image `/api/agent/v1/file`) are
**not** token-gated in v1 — if you expose this backend cross-machine, bind the UI
to localhost or add auth there (follow-up).

### Interactive conversation — `/api/agent/v1/tools/workspaces*`

```bash
# 1. create a durable workspace, then a conversation inside it
workspace_id=$(curl -sS http://127.0.0.1:8765/api/agent/v1/tools/workspaces \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $VIBESIM_API_TOKEN" \
  -d '{"displayName":"Llama 3 H200 study"}' \
  | uv run python -c 'import sys,json;print(json.load(sys.stdin)["workspace_id"])')
cid=$(curl -sS "http://127.0.0.1:8765/api/agent/v1/tools/workspaces/$workspace_id/conversations" \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $VIBESIM_API_TOKEN" \
  -d '{"sandbox":"workspace-write"}' \
  | uv run python -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# 2. send a turn; read `final`. If it is a question or you want to steer, send another.
curl -sS "http://127.0.0.1:8765/api/agent/v1/tools/workspaces/$workspace_id/conversations/$cid/messages" \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $VIBESIM_API_TOKEN" \
  -d '{"text":"Simulate Llama-3-8B dense on 1xH200 at Poisson rate 48; report throughput and TPOT."}'

# 3. retain $cid for human observation and artifact retrieval
echo "VibeSim workspace/conversation: $workspace_id / $cid"
```

A turn is **synchronous** and may take minutes (profiling/sim) — set a generous
client timeout of at least 30 minutes and wait for the same request to return;
do not replace it with manual GET polling. The turn response includes `final`,
`ok`, `conversation_id`,
`implementer_summaries`, `intermediate_outputs`, `tool_calls`, and `sessions`.
Each intermediate output has `level: progress | milestone`; `tool_calls` is the
separate transient command/container/tool activity channel.
`scripts/agent_conversation_smoke.sh` exercises this whole path.

Calling agents must leave conversations intact on success and failure so a
human can inspect progress and artifacts. The DELETE endpoint is reserved for
explicit human/operator cleanup.

### Single-turn eval — `POST /api/agent/v1/tools/eval` (evaluation only)

For capability checks / testcases that do not need a conversation. **Not the
interactive interface** — prefer `/api/agent/v1/tools/workspaces*` for real agent work.

```bash
curl -sS http://127.0.0.1:8765/api/agent/v1/tools/eval \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $VIBESIM_API_TOKEN" \
  -d '{"prompt":"List the available VibeSim L1 profilers."}'
```

`/api/agent/v1/tools/eval` prepares the isolated workspace and Docker Codex container. It does
not write to the UI conversation list, does not use SSE, and defaults to
`autonomous: true`. It is **synchronous** — the response returns only after the
task finishes, so set a generous client timeout for profiling/sim runs. By
default it keeps the eval workspace so code, logs, plots, and artifacts can be
fetched afterwards (via `/api/agent/v1/tools/workspaces/{workspace_id}/artifacts*`),
but removes the Docker container after the run. Batch execution is intentionally
outside the backend: run multiple `/api/agent/v1/tools/eval` calls from the harness with the
concurrency you want.

Useful request fields:

- `prompt`: required user task.
- `sandbox`: optional, default `workspace-write`.
- `autonomous`: optional, default `true`.
- `keep_container`: optional, default `false`; normally leave this off so evals
  do not accumulate Docker containers.

The response includes `workspace_id`, `conversation_id`, `final`, `ok`,
`implementer_summaries`, `intermediate_outputs`, `tool_calls`, and `workspace`.

### Artifact retrieval

```bash
# list files the run produced
curl -sS -G http://127.0.0.1:8765/api/agent/v1/tools/workspaces/w_eval-1a2b3c4d5e6f/artifacts \
  -H "Authorization: Bearer $VIBESIM_API_TOKEN" \
  --data-urlencode "subdir=logs"

# download one of them
curl -sS -OJ -G http://127.0.0.1:8765/api/agent/v1/tools/workspaces/w_eval-1a2b3c4d5e6f/artifacts/download \
  -H "Authorization: Bearer $VIBESIM_API_TOKEN" \
  --data-urlencode "path=logs/<run>/summary.json"
```

`scripts/agent_api_smoke.sh` exercises this whole path end to end.

## Debug Logging

The backend writes JSON-line logs to `logs/backend.log` and stdout. Useful
fields:

- `conversation_id`, `turn_id` — correlate one browser turn across the backend;
- `prompt_fingerprint` — changes when role prompts/schema/model change;
- `role`, `resume`, `codex_session_id` — confirm orchestrator/implementer
  routing and Codex resume behavior;
- `orchestrator.decision.action` — shows whether the orchestrator answered the
  user directly or delegated to the implementer;
- `container`, `workspace` — confirm which isolated Docker workspace is active;
- `final_preview`, `stderr_tail`, `returncode` — debug Codex CLI failures.

## Execution Modes

- `read-only`: the backend will not run the implementer. The orchestrator can
  answer, ask, or tell the user that write mode is needed.
- `workspace-write`: the implementer can edit and run commands inside the copied
  Docker workspace.
- `danger-full-access`: same copied workspace and bypassed Codex sandboxing.

The Autonomous button is independent from execution mode. It is available in the
welcome area before the first user message, and changes the workspace prompt
file, not filesystem permissions: the driving role should avoid clarification
questions and continue with stated assumptions, while still stopping for missing
credentials or destructive/shared-state authorization.

Note that `sandbox` never reaches the Codex CLI — `codex_command.py` always
passes `--dangerously-bypass-approvals-and-sandbox`, so `read-only`'s one
enforced effect is that `turn.py` refuses to delegate. In `agent_mode=single`
there is nothing to delegate, so `read-only` degrades to a prompt-only
constraint stated in the single-mode `AGENTS*.md`. This is a pre-existing gap in
how `sandbox` is wired, not something single mode introduced; treat `read-only`
as advisory in that mode.

GPU forwarding is controlled independently by `CODEX_DOCKER_GPUS`. It defaults
to `all`, so `workspace-write` containers can run CUDA smoke checks and
profiling code inside the copied workspace. Set `CODEX_DOCKER_GPUS=` to disable
Docker GPU forwarding.

## Layout

| Path                                        | Purpose                                                                                                  |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `backend/app.py`                            | FastAPI routes + SSE streaming + Vite static serving                                                     |
| `backend/codex_runtime/config.py`           | environment, path, mode, prompt-fingerprint settings                                                     |
| `backend/codex_runtime/workspace.py`        | per-workspace `VibeSim/` copy and local git bootstrap                                                       |
| `backend/codex_runtime/docker.py`           | Docker container lifecycle and isolated Codex home setup                                                 |
| `backend/codex_runtime/exec_types.py`       | shared Codex execution request/event types                                                               |
| `backend/codex_runtime/codex_command.py`    | Docker + `codex exec` command construction                                                               |
| `backend/codex_runtime/codex_cli.py`        | Codex subprocess lifecycle, timeout, and cancellation                                                    |
| `backend/codex_runtime/output_collector.py` | stdout/stderr/rollout collection into UI events                                                          |
| `backend/codex_runtime/codex_events.py`     | Codex JSON/rollout event translation                                                                     |
| `backend/codex_runtime/prompts.py`          | role prompts and orchestrator JSON parsing                                                               |
| `backend/codex_runtime/turn.py`             | high-level turn loop for both agent modes                                                                |
| `backend/analyzer_evidence_mcp/server.py`   | bounded read-only MCP bridge to the Analyzer `/api/v1/*` resources                                       |
| `backend/naming.py`                         | non-blocking OpenRouter structured naming plus pending-state scheduling                                  |
| `backend/eval.py`                           | JSON `/api/agent/v1/tools/eval` wrapper around one `run_turn()` (+ shared `collect_turn_event`)                         |
| `backend/artifacts.py`                      | list/resolve files in a run workspace for the agent artifact endpoints                                   |
| `SKILL.md`                                  | agent skill (capabilities, when-to-call, what-to-expect, HTTP contract) served at `GET /api/agent/v1/tools/skill` |
| `backend/prompt_templates/*.j2`             | Jinja2 sources for the generated prompts; edit these, never `backend/prompts/`                           |
| `backend/agents_prompt.py`                  | prompt renderer; `ensure_rendered()` runs on `codex_runtime/config` import                               |
| `backend/prompts/AGENTS*.md`                | generated (gitignored) agent_mode × autonomous matrix, one mounted read-only as `/workspace/AGENTS.md`   |
| `backend/prompts/*.txt`                     | short role startup prompts; orchestrator/assistant generated, implementer hand-written                   |
| `backend/prompts/*.schema.json`             | decision-envelope schemas passed to `codex exec --output-schema`                                         |
| `backend/store.py`                          | workspace registry plus per-workspace SQLite conversation/event store                                    |
| `docker/codex-runner.Dockerfile`            | prebuilt CUDA runner image with Node, Codex CLI, `uv`, git, Rust, `just`, `nvcc`, and baked VibeSim deps |
| `scripts/build-codex-runner-image.sh`       | one-shot image builder used by `run.sh` when needed                                                      |
| `frontend/`                                 | React + TypeScript + Vite chat UI                                                                        |
| `../agent-workspaces/`                      | generated workspace envelopes, shared repos, Codex homes, and SQLite state                               |

## Environment

- `VIBESIM_API_TOKEN` — bearer token gating the agent endpoints (`/api/agent/v1/tools/eval`,
  `/api/agent/v1/tools/workspaces*`). Unset → those endpoints are open
  (local dev). Set → they require `Authorization: Bearer <token>`. `/api/agent/v1/tools/skill`
  is public regardless.
- `CODEX_MODEL` — default model of the `gpt` family, default `gpt-5.6-sol`.
- `CODEX_REASONING_EFFORT` — default reasoning effort of the `gpt` family, passed
  per call as `-c model_reasoning_effort=...`, default `xhigh`.
- `CODEXDS_MODEL` / `CODEXDS_REASONING_EFFORT` — the same two defaults for the
  `deepseek` family, default `deepseek-ai/DeepSeek-V4-Flash-0731` and `max`.
- `CODEX_TRADITIONAL_HOME` / `CODEXDS_HOME` — each family's host Codex home
  (auth profile plus model catalog), default `~/.codex` and `~/.codex-ds`.
- `CODEX_DOCKER_IMAGE` — Docker image. By default it uses the current OS user as
  a stable tag, for example `vibesim-ui-codex-runner:kanzhu`. This prevents one
  user on a shared Docker host from replacing another user's UID/GID-specific
  runner image. Override the same value for both image build and backend startup
  when a deployment needs a different tag.
- `CODEX_CUDA_IMAGE` — CUDA devel base image baked into the runner image,
  default `nvidia/cuda:12.8.1-devel-ubuntu24.04`.
- `CODEX_UV_IMAGE` — source image copied for the `uv`/`uvx` binaries, default
  `ghcr.io/astral-sh/uv:python3.12-bookworm`.
- `CODEX_NPM_PACKAGE` — Codex npm package baked into the image by the build
  script, default `@openai/codex@0.144.0`.
- `RUST_TOOLCHAIN` — Rust toolchain baked into the image by the build script,
  default `stable`.
- `CODEX_RUNNER_IMAGE_VERSION` — expected image label, default
  `prebuilt-agent-runner-v11`. `run.sh` rebuilds when this label differs or when
  the baked `VibeSim/uv.lock` hash differs. Ordinary VibeSim source changes do not
  rebuild the image; Cargo compiles first-party crates inside each workspace
  against the dependency-only target seed.
- `CODEX_SKIP_RUNNER_IMAGE_TEST=1` — skip the post-build non-GPU runner
  acceptance gate for an explicitly incomplete development build.
- `CODEX_FORCE_IMAGE_BUILD=1` — force `run.sh` to rebuild the runner image.
- `CODEX_SKIP_IMAGE_BUILD=1` — skip the image existence check/build step.
- `CODEX_IDLE_TIMEOUT` — per Codex call idle timeout in seconds, default `600`.
  Long tasks may run past this as long as Codex keeps producing stdout or
  rollout commentary. `CODEX_TURN_TIMEOUT` is still accepted as a backward
  compatible fallback name.
- `CODEX_DOCKER_GPUS` — value passed to `docker run --gpus`, default `all`.
  Set it to an empty string to run without GPU forwarding.
- `CODEX_DOCKER_DG_USE_LOCAL_VERSION` — DeepGEMM build mode inside Docker,
  default `0`. This is the repo-supported DeepGEMM path from `VibeSim/justfile`;
  it keeps DeepGEMM enabled while avoiding install-time build-clone assertions.
- `CODEX_DOCKER_UID`, `CODEX_DOCKER_GID`, `CODEX_DOCKER_USER`,
  `CODEX_DOCKER_HOME` — optional container identity override. Defaults to the
  host user, so files written under `/workspace` are not root-owned and Codex
  sees the same absolute `.codex` home path.
- `CODEX_DOCKER_UV_PROJECT_ENVIRONMENT` — where `uv run` creates the project
  virtualenv inside Docker, default `/opt/vibesim-venv`. The default is baked into
  the runner image and chowned to the host UID/GID; per-container overlay writes
  are isolated from other conversations.
- `CODEX_DOCKER_UV_CACHE_DIR` — where `uv` stores cache inside Docker, default
  `/opt/vibesim-uv-cache`, also baked into the runner image.
- `VIBESIM_WORKSPACES_ROOT` — shared workspace registry/state root, default
  `../agent-workspaces`.
- `OPENROUTER_API_KEY` — enables non-blocking automatic naming for new UI
  workspaces and conversations. Unset disables the request and keeps fallback
  names pending for a later successful turn. `OPENROUTE_KEY` is accepted as a
  compatibility alias for the existing host secret; `OPENROUTER_API_KEY` takes
  precedence when both are set.
- `VIBESIM_NAMING_MODEL` — OpenRouter model used for naming, default
  `deepseek/deepseek-v4-flash`.
- `VIBESIM_NAMING_BASE_URL` — OpenRouter-compatible API root, default
  `https://openrouter.ai/api/v1`.
- `VIBESIM_NAMING_TIMEOUT_SECONDS` — naming request timeout, default `8`.
  Requests require structured output and zero-data-retention routing.
- `VIBESIM_MANAGED_BACKEND_URL` — callback origin written into short-lived
  managed Launcher capabilities, default
  `http://host.docker.internal:8765`. It must resolve from the Codex container.
- `ANALYZER_MCP_SOURCE` — default Analyzer transport mode, `external`. The MCP
  tool exposes the clearer per-call names `source="host"` for the shared host
  Analyzer and `source="workspace"` for a workspace-local Analyzer. Source
  chooses where data is read; citation authorization remains workspace-scoped,
  so a ready result from another conversation in the same workspace is valid.
- `ANALYZER_MCP_BASE_URL` — host Analyzer origin, default
  `http://host.docker.internal:8787`. Bind the host Analyzer only to the Docker
  bridge address rather than all interfaces. The container receives an
  explicit `host.docker.internal:host-gateway` mapping.
- `ANALYZER_MCP_LOGS_ROOT` and `ANALYZER_MCP_REPO_ROOT` — local-mode paths,
  default `/workspace/logs` and `/workspace`.
- `HF_HOME` — optional host Hugging Face cache directory. When set, every
  conversation container bind-mounts it read-only at `/model` and receives
  `HF_HOME=/model`. A configured path must already exist.
- `FRONTEND_SKIP_BUILD=1` — skip `npm ci` / `npm run build` in `run.sh`, useful
  when `npm run dev` is serving the frontend separately.
- `PORT`, `HOST` — FastAPI bind settings used by `run.sh`.

This is a local development tool. It copies host `~/.codex` authentication and
configuration into a conversation-specific `codex-home` inside the selected
workspace, then bind-mounts that clean home into Docker as
`/home/<user>/.codex`. Conversations share the workspace repo and experiments,
while `tmp`, sessions, and rollout logs stay isolated per conversation.

### Claude startup credential discovery

`run.sh` launches the backend through `scripts/with_claude_env.py`. Existing
nonempty authentication environment variables take precedence. Otherwise the
launcher reads simple `claude` alias/function definitions from local shell
startup files (`.bash_profile`, `.profile`, `.bashrc`, `.bash_aliases`, `.zshrc`).
It recognizes leading literal assignments and inherited `$VAR`/`${VAR}` values
followed by `claude` or `command claude "$@"`. It does not source these files,
execute wrappers, follow sourced files, or support arbitrary shell code.
Only ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, CLAUDE_CODE_OAUTH_TOKEN and
ANTHROPIC_BASE_URL are imported. An explicitly configured authentication method
never inherits a different wrapper's endpoint or token. A mismatched explicit
base URL also prevents fallback. No credential values are logged or saved.
Set `CLAUDE_DISCOVER_SHELL_ENV=0` to disable discovery. For custom launchers use:

```bash
python3 scripts/with_claude_env.py uv run --frozen uvicorn backend.app:app
```

Restart the backend after changing shell definitions. This enables the menu
when credentials are found; provider validity still requires a real request.
