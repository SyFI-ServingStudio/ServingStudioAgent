# user-facing-ui

A small web chat UI for VibeSim. The browser talks to a FastAPI backend, which
drives **`codex exec` inside Docker**.

Each conversation gets an isolated copy of git-tracked files from `../main`.
That copy is mounted read/write into Docker at `/workspace`, while the real
`../main` tree is left untouched.

## Run

```bash
cd user-facing-ui
./run.sh
# then open http://<host>:8765
```

`run.sh` first builds the React/Vite frontend into `frontend/dist`, then starts
the FastAPI backend. If `frontend/node_modules` is missing, it runs `npm ci`
once before the build. Set `FRONTEND_SKIP_BUILD=1` when you are already running
the Vite dev server.

The backend uses a prebuilt local Docker image for the Codex runner. If the image
is missing, its VibeSim runner label is stale, or its baked `main/uv.lock` hash
does not match the current checkout, `run.sh` builds it once from
`docker/codex-runner.Dockerfile`; later turns and later conversations reuse that
image. The image is based on CUDA 12.8 devel and includes Node/Codex, `uv`, git,
Rust stable (`cargo`/`rustc`), `just`, `nvcc`, Python 3.12 dev headers, native
build tools, and a prewarmed VibeSim Python environment at `/opt/vibesim-venv`.
The image build copies the tracked `main` tree, prewarms the default profiling
stack (including pinned DeepGEMM), and compiles the full Cargo release target for
the simulator and analyzer. That target is stored at
`/opt/vibesim-cache/target`. A new conversation copies this seed into its empty
`/workspace/target` before Codex starts, so the launcher's ordinary Cargo build
still performs its freshness checks while reusing the expensive dependency
artifacts. The analyzer intentionally tracks the workspace Git HEAD for embedded
provenance, so a fresh conversation may rerun its build script and final link;
it should not recompile DataFusion/Arrow from scratch. The source is compiled at
the runtime path `/workspace`, and the image is invalidated by a fingerprint of
the tracked Cargo/simulator/analyzer inputs.
The Docker container and `codex exec` both run as the host UID/GID with `HOME`
set to the matching `/home/<user>` path.

To rebuild the runner image explicitly:

```bash
cd user-facing-ui
CODEX_FORCE_IMAGE_BUILD=1 ./run.sh
```

### Runner image acceptance test

After building the image, run the non-GPU acceptance test:

```bash
cd user-facing-ui
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

Frontend-only development can run Vite against the same backend:

```bash
cd user-facing-ui
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
  -> prepare workspaces/<conversation-id>/main from git-tracked ../main files
     - copy backend/prompts/AGENTS.md or AGENTS.autonomous.md to /workspace/AGENTS.md
     - link /workspace/.codex/skills -> /workspace/skills
  -> seed workspaces/<id>/codex-home from host ~/.codex auth/config
  -> docker run -v workspaces/<id>/main:/workspace -v workspaces/<id>/codex-home:/home/<user>/.codex
     - when host HF_HOME is set, mount it read-only at /model and set container HF_HOME=/model
     using the prebuilt CODEX_DOCKER_IMAGE
  -> codex exec/resume as orchestrator
       action=user_message   -> return ask/notify text to the user
       action=run_implementer -> codex exec/resume as implementer
  -> explicit handoff of implementer summary back to orchestrator
       action=user_message   -> return reviewed result to the user
       action=run_implementer -> continue with another bounded implementer task
  -> retain the active turn independently of the browser connection
       GET .../stream -> replay and continue after refresh
       POST .../cancel -> send SIGINT to Codex and stop the whole turn
  -> stream progress + final text back to browser
```

The orchestrator is normally an active human-in-the-loop coordinator. It reads
matching skills, classifies the request, decides whether clarification is
needed, and delegates only bounded implementer tasks. Before the first message
in a conversation, the welcome area shows an Autonomous button below the example
questions. When enabled, the backend copies
`backend/prompts/AGENTS.autonomous.md` instead; that prompt tells the
orchestrator to proceed with conservative assumptions instead of asking
preference or clarification questions. After the first user message, the
conversation's autonomous setting is fixed.

The implementer returns free-form text; there is no judge, profiler, or shared
`profile.db` write unless the copied workspace task does it. The orchestrator
and implementer keep separate Codex session ids. Same-role continuity uses
`codex exec resume`; cross-role handoff does not rely on shared context and is
passed explicitly as task text and implementer summary.
`backend/prompts/AGENTS.md` and `backend/prompts/AGENTS.autonomous.md` hold the
detailed shared role and skill instructions. One of them is copied into the
workspace as `/workspace/AGENTS.md`. The workspace also gets `.codex/skills ->
../skills` so Codex can discover the copied repo-local skills. The role startup
prompts are sent only when a role session is first created; later turns resume
that role and send only the new user message or delegated task. Implementer
summaries are explicitly sent back to the orchestrator before the turn finishes.

The UI shows assistant intermediate output and, when work is delegated, the
implementer summary. If the orchestrator delegates multiple follow-ups in one
browser turn, the final assistant message includes the implementer summaries and
the orchestrator's final user-facing message.

Long browser conversations load backwards in fixed-size message pages. The
browser requests the newest page with
`GET /api/conversations/{cid}?limit=<n>` and requests an older page with
`?limit=<n>&before=<start_index>`, where `start_index` comes from the current
response's `message_page`. Reaching the top of the message viewport triggers the
older request and preserves the visible scroll position while prepending it.
Calling the same endpoint without query parameters retains the original
full-history response. Pagination never trims stored messages or changes role
session continuity.

## Agent API

Besides the browser UI, VibeSim exposes a small **HTTP surface for other agents**
to call. It is self-describing: fetch `GET /api/agent/skill` to get the full skill
(`SKILL.md` — what VibeSim does, when to call it, what to expect, and the
contract), then drive everything with plain HTTP — no framework glue.

The **real interactive interface** is the agent conversation API: multi-turn,
synchronous JSON, with workspace + Codex-session continuity across turns (it
reuses the same `store` and `run_turn` as the browser SSE path). The calling
agent reads each turn's `final` and, like a human, answers clarifying questions
or steers with another turn. `/api/eval` is **evaluation-only** (single-turn,
stateless; for testcases).

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/api/agent/skill` | public | Agent skill doc (`SKILL.md`, `text/markdown`). |
| POST | `/api/agent/conversations` | token | Create an interactive conversation. |
| POST | `/api/agent/conversations/{cid}/messages` | token | Run one turn; synchronous JSON. |
| GET | `/api/agent/conversations/{cid}` | token | Full conversation history. |
| DELETE | `/api/agent/conversations/{cid}` | token | Human/operator cleanup only; calling agents must not invoke it. |
| GET | `/api/agent/artifacts` | token | List files in a conversation's workspace. |
| GET | `/api/agent/artifacts/download` | token | Download one workspace file. |
| POST | `/api/eval` | token | Single-turn evaluation only (not interactive). |

**Auth** is gated by `VIBESIM_API_TOKEN`. When it is set, agent endpoints require
`Authorization: Bearer <token>` (missing/wrong → `401`); `/api/agent/skill` stays
public. When it is unset (local dev / same-host eval harness), no header is
needed. The browser UI routes (`/api/conversations*`, image `/api/file`) are
**not** token-gated in v1 — if you expose this backend cross-machine, bind the UI
to localhost or add auth there (follow-up).

### Interactive conversation — `/api/agent/conversations*`

```bash
# 1. create (autonomous defaults false, so VibeSim will ask clarifying questions)
cid=$(curl -sS http://127.0.0.1:8765/api/agent/conversations \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $VIBESIM_API_TOKEN" \
  -d '{"sandbox":"workspace-write"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# 2. send a turn; read `final`. If it is a question or you want to steer, send another.
curl -sS "http://127.0.0.1:8765/api/agent/conversations/$cid/messages" \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $VIBESIM_API_TOKEN" \
  -d '{"text":"Simulate Llama-3-8B dense on 1xH200 at Poisson rate 48; report throughput and TPOT."}'

# 3. retain $cid for human observation and artifact retrieval
echo "VibeSim conversation: $cid"
```

A turn is **synchronous** and may take minutes (profiling/sim) — set a generous
client timeout of at least 30 minutes and wait for the same request to return;
do not replace it with manual GET polling. The turn response includes `final`,
`ok`, `conversation_id`,
`implementer_summaries`, `intermediate_outputs`, `progress`, and `sessions`.
`scripts/agent_conversation_smoke.sh` exercises this whole path.

Calling agents must leave conversations intact on success and failure so a
human can inspect progress and artifacts. The DELETE endpoint is reserved for
explicit human/operator cleanup.

### Single-turn eval — `POST /api/eval` (evaluation only)

For capability checks / testcases that do not need a conversation. **Not the
interactive interface** — prefer `/api/agent/conversations*` for real agent work.

```bash
curl -sS http://127.0.0.1:8765/api/eval \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $VIBESIM_API_TOKEN" \
  -d '{"prompt":"List the available VibeSim L1 profilers."}'
```

`/api/eval` prepares the isolated workspace and Docker Codex container. It does
not write to the UI conversation list, does not use SSE, and defaults to
`autonomous: true`. It is **synchronous** — the response returns only after the
task finishes, so set a generous client timeout for profiling/sim runs. By
default it keeps the eval workspace so code, logs, plots, and artifacts can be
fetched afterwards (via `/api/agent/artifacts*` using the returned `conversation_id`),
but removes the Docker container after the run. Batch execution is intentionally
outside the backend: run multiple `/api/eval` calls from the harness with the
concurrency you want.

Useful request fields:

- `prompt`: required user task.
- `sandbox`: optional, default `workspace-write`.
- `autonomous`: optional, default `true`.
- `keep_container`: optional, default `false`; normally leave this off so evals
  do not accumulate Docker containers.

The response includes `conversation_id` (use as `cid` for artifact retrieval),
`final`, `ok`, `implementer_summaries`, `progress`, and `workspace`.

### Artifact retrieval

```bash
# list files the run produced
curl -sS -G http://127.0.0.1:8765/api/agent/artifacts \
  -H "Authorization: Bearer $VIBESIM_API_TOKEN" \
  --data-urlencode "cid=eval-1a2b3c4d5e6f" --data-urlencode "subdir=logs"

# download one of them
curl -sS -OJ -G http://127.0.0.1:8765/api/agent/artifacts/download \
  -H "Authorization: Bearer $VIBESIM_API_TOKEN" \
  --data-urlencode "cid=eval-1a2b3c4d5e6f" \
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
file, not filesystem permissions: the orchestrator should avoid clarification
questions and continue with stated assumptions, while still stopping for missing
credentials or destructive/shared-state authorization.

GPU forwarding is controlled independently by `CODEX_DOCKER_GPUS`. It defaults
to `all`, so `workspace-write` containers can run CUDA smoke checks and
profiling code inside the copied workspace. Set `CODEX_DOCKER_GPUS=` to disable
Docker GPU forwarding.

## Layout

| Path | Purpose |
|------|---------|
| `backend/app.py` | FastAPI routes + SSE streaming + Vite static serving |
| `backend/codex_runtime/config.py` | environment, path, mode, prompt-fingerprint settings |
| `backend/codex_runtime/workspace.py` | per-conversation `main/` copy and local git bootstrap |
| `backend/codex_runtime/docker.py` | Docker container lifecycle and isolated Codex home setup |
| `backend/codex_runtime/exec_types.py` | shared Codex execution request/event types |
| `backend/codex_runtime/codex_command.py` | Docker + `codex exec` command construction |
| `backend/codex_runtime/codex_cli.py` | Codex subprocess lifecycle, timeout, and cancellation |
| `backend/codex_runtime/output_collector.py` | stdout/stderr/rollout collection into UI events |
| `backend/codex_runtime/codex_events.py` | Codex JSON/rollout event translation |
| `backend/codex_runtime/prompts.py` | role prompts and orchestrator JSON parsing |
| `backend/codex_runtime/turn.py` | high-level orchestrator/implementer turn loop |
| `backend/eval.py` | JSON `/api/eval` wrapper around one `run_turn()` (+ shared `collect_turn_event`) |
| `backend/artifacts.py` | list/resolve files in a run workspace for the agent artifact endpoints |
| `SKILL.md` | agent skill (capabilities, when-to-call, what-to-expect, HTTP contract) served at `GET /api/agent/skill` |
| `backend/prompts/AGENTS.md` | detailed instructions copied into each `/workspace` |
| `backend/prompts/AGENTS.autonomous.md` | autonomous-mode instructions copied as `/workspace/AGENTS.md` |
| `backend/prompts/*.txt` | short role startup prompts for orchestrator/implementer |
| `backend/store.py` | in-memory + JSON-file conversation store |
| `docker/codex-runner.Dockerfile` | prebuilt CUDA runner image with Node, Codex CLI, `uv`, git, Rust, `just`, `nvcc`, and baked VibeSim deps |
| `scripts/build-codex-runner-image.sh` | one-shot image builder used by `run.sh` when needed |
| `frontend/` | React + TypeScript + Vite chat UI |
| `workspaces/` | generated per-conversation copies of `../main` |

## Environment

- `VIBESIM_API_TOKEN` — bearer token gating the agent endpoints (`/api/eval`,
  `/api/agent/artifacts`, `/api/agent/artifacts/download`). Unset → those endpoints are open
  (local dev). Set → they require `Authorization: Bearer <token>`. `/api/agent/skill`
  is public regardless.
- `CODEX_MODEL` — Codex model, default `gpt-5.6-sol`.
- `CODEX_REASONING_EFFORT` — Codex reasoning effort passed as
  `-c model_reasoning_effort=...`, default `xhigh`.
- `CODEX_DOCKER_IMAGE` — Docker image, default `vibesim-ui-codex-runner:latest`.
- `CODEX_CUDA_IMAGE` — CUDA devel base image baked into the runner image,
  default `nvidia/cuda:12.8.1-devel-ubuntu24.04`.
- `CODEX_UV_IMAGE` — source image copied for the `uv`/`uvx` binaries, default
  `ghcr.io/astral-sh/uv:python3.12-bookworm`.
- `CODEX_NPM_PACKAGE` — Codex npm package baked into the image by the build
  script, default `@openai/codex@0.144.0`.
- `RUST_TOOLCHAIN` — Rust toolchain baked into the image by the build script,
  default `stable`.
- `CODEX_RUNNER_IMAGE_VERSION` — expected image label, default
  `prebuilt-codex-runner-v8`. `run.sh` rebuilds when this label differs, when
  the baked `main/uv.lock` hash differs, or when the tracked Cargo workspace,
  simulator, or analyzer build-input fingerprint differs from the checkout.
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
  default `0`. This is the repo-supported DeepGEMM path from `main/justfile`;
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
- `HF_HOME` — optional host Hugging Face cache directory. When set, every
  conversation container bind-mounts it read-only at `/model` and receives
  `HF_HOME=/model`. A configured path must already exist.
- `FRONTEND_SKIP_BUILD=1` — skip `npm ci` / `npm run build` in `run.sh`, useful
  when `npm run dev` is serving the frontend separately.
- `PORT`, `HOST` — FastAPI bind settings used by `run.sh`.

This is a local development tool. It copies host `~/.codex` authentication and
configuration into an isolated per-conversation `codex-home`, then bind-mounts
that clean home into Docker as `/home/<user>/.codex`. Runtime state such as
`tmp`, `sessions`, and rollout logs stays isolated per conversation.
