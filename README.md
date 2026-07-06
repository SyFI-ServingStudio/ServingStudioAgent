# user-facing-ui

A small web chat UI for MLSim. The browser talks to a FastAPI backend, which
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
is missing, its MLSim runner label is stale, or its baked `main/uv.lock` hash
does not match the current checkout, `run.sh` builds it once from
`docker/codex-runner.Dockerfile`; later turns and later conversations reuse that
image. The image is based on CUDA 12.8 devel and includes Node/Codex, `uv`, git,
Rust stable (`cargo`/`rustc`), `just`, `nvcc`, Python 3.12 dev headers, native
build tools, and a prewarmed MLSim Python environment at `/opt/mlsim-venv`.
The prewarmed env is built from `../main/pyproject.toml`, `../main/uv.lock`,
and `../main/justfile`, so the default profiling stack, including pinned
DeepGEMM, is already installed before any conversation starts. The Docker
container and `codex exec` both run as the host UID/GID with `HOME` set to the
matching `/home/<user>` path.

To rebuild the runner image explicitly:

```bash
cd user-facing-ui
CODEX_FORCE_IMAGE_BUILD=1 ./run.sh
```

Frontend-only development can run Vite against the same backend:

```bash
cd user-facing-ui
./run.sh
# in another shell
cd frontend
npm run dev
# then open http://<host>:5173
```

## Turn Flow

```text
browser
  -> FastAPI
  -> prepare workspaces/<conversation-id>/main from git-tracked ../main files
     - copy backend/prompts/AGENTS.md or AGENTS.autonomous.md to /workspace/AGENTS.md
     - link /workspace/.codex/skills -> /workspace/skills
  -> seed workspaces/<id>/codex-home from host ~/.codex auth/config
  -> docker run -v workspaces/<id>/main:/workspace -v workspaces/<id>/codex-home:/home/<user>/.codex
     using the prebuilt CODEX_DOCKER_IMAGE
  -> codex exec/resume as orchestrator
       action=user_message   -> return ask/notify text to the user
       action=run_implementer -> codex exec/resume as implementer
  -> explicit handoff of implementer summary back to orchestrator
       action=user_message   -> return reviewed result to the user
       action=run_implementer -> continue with another bounded implementer task
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

## Eval API

For capability checks that do not need the browser, call the single-turn JSON
endpoint:

```bash
curl -sS http://127.0.0.1:8765/api/eval \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"List the available MLSim L1 profilers."}'
```

`/api/eval` still prepares the isolated workspace and Docker Codex container.
It does not write to the UI conversation list, does not use SSE, and defaults to
`autonomous: true`. By default it keeps the temporary eval workspace so code,
logs, plots, and artifacts can be inspected, but removes the corresponding
Docker container after the run. Batch execution is intentionally outside the
backend: run multiple `/api/eval` calls from the test harness with the
concurrency you want.

Useful request fields:

- `prompt`: required user task.
- `sandbox`: optional, default `workspace-write`.
- `autonomous`: optional, default `true`.
- `keep_container`: optional, default `false`; normally leave this off so evals
  do not accumulate Docker containers.

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
| `backend/eval.py` | JSON `/api/eval` wrapper around one `run_turn()` |
| `backend/prompts/AGENTS.md` | detailed instructions copied into each `/workspace` |
| `backend/prompts/AGENTS.autonomous.md` | autonomous-mode instructions copied as `/workspace/AGENTS.md` |
| `backend/prompts/*.txt` | short role startup prompts for orchestrator/implementer |
| `backend/store.py` | in-memory + JSON-file conversation store |
| `docker/codex-runner.Dockerfile` | prebuilt CUDA runner image with Node, Codex CLI, `uv`, git, Rust, `just`, `nvcc`, and baked MLSim deps |
| `scripts/build-codex-runner-image.sh` | one-shot image builder used by `run.sh` when needed |
| `frontend/` | React + TypeScript + Vite chat UI |
| `workspaces/` | generated per-conversation copies of `../main` |

## Environment

- `CODEX_MODEL` — Codex model, default `gpt-5.3-codex-spark`.
- `CODEX_DOCKER_IMAGE` — Docker image, default `mlsim-ui-codex-runner:latest`.
- `CODEX_CUDA_IMAGE` — CUDA devel base image baked into the runner image,
  default `nvidia/cuda:12.8.1-devel-ubuntu24.04`.
- `CODEX_UV_IMAGE` — source image copied for the `uv`/`uvx` binaries, default
  `ghcr.io/astral-sh/uv:python3.12-bookworm`.
- `CODEX_NPM_PACKAGE` — Codex npm package baked into the image by the build
  script, default `@openai/codex@0.125.0`.
- `RUST_TOOLCHAIN` — Rust toolchain baked into the image by the build script,
  default `stable`.
- `CODEX_RUNNER_IMAGE_VERSION` — expected image label, default
  `prebuilt-codex-runner-v5`. `run.sh` rebuilds when this label differs or
  when the baked `main/uv.lock` hash label differs from the current checkout.
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
  virtualenv inside Docker, default `/opt/mlsim-venv`. The default is baked into
  the runner image and chowned to the host UID/GID; per-container overlay writes
  are isolated from other conversations.
- `CODEX_DOCKER_UV_CACHE_DIR` — where `uv` stores cache inside Docker, default
  `/opt/mlsim-uv-cache`, also baked into the runner image.
- `FRONTEND_SKIP_BUILD=1` — skip `npm ci` / `npm run build` in `run.sh`, useful
  when `npm run dev` is serving the frontend separately.
- `PORT`, `HOST` — FastAPI bind settings used by `run.sh`.

This is a local development tool. It copies host `~/.codex` authentication and
configuration into an isolated per-conversation `codex-home`, then bind-mounts
that clean home into Docker as `/home/<user>/.codex`. Runtime state such as
`tmp`, `sessions`, and rollout logs stays isolated per conversation.
