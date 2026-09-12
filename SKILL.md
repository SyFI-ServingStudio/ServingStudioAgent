---
name: use-vibesim
description: Use the hosted ServingStudio Agent API for ML-serving simulation, timing prediction, kernel/GPU queries, analyzer work, model exploration, and simulator extension. Use when an external agent needs a durable ServingStudio Sim workspace, one or more multi-turn conversations, generated simulation artifacts, or Analyzer-backed performance evidence.
---

# ServingStudio Sim — Agent Skill

**This is a skill for other agents.** It tells you *what ServingStudio Sim can do*, *when to
reach for it*, *what to expect when you do*, and *how to call it* over plain HTTP.
It is served live at `GET /api/agent/v1/tools/skill`, so you can fetch this one URL and start
driving ServingStudio Sim with no framework-specific glue.

ServingStudio Sim (a.k.a. ServingStudio Sim) is a **discrete-event simulator for ML serving/training
workloads**. It predicts the performance of an LLM inference *deployment* from
**measured GPU kernel costs** — not from a real serving run. You do not call
low-level functions; you talk to an interactive assistant that plans the work,
runs it inside an isolated sandbox, asks you clarifying questions when needed,
and returns a written answer plus any artifacts (logs, JSON, plots) it produced.

---

## 1. What ServingStudio Sim can do

| Capability | What it gives you | Typical ask |
|---|---|---|
| **Simulate a deployment** | Throughput, TTFT, TPOT, GPU utilization for a full run from a preset + workload. Covers dense **and** MoE models; TP/EP/PP parallelism; PD (prefill/decode) and AFD (attention/FFN) disaggregation. | "Simulate Llama-3-8B dense on 1×H200 under a Poisson workload at rate 48 and report throughput and TPOT." |
| **Predict per-iteration cost** | Cost of one explicit batch shape at fixed current KV lengths *without* prefill progression, the scheduler/clock, or a workload. | "Predict the per-iteration decode cost for Llama-3-8B on H200 at batch=256, kv_len=4096." |
| **Profile a kernel** | Measured latency/throughput of a real kernel on a real GPU, from `profile.db`. | "On H200, bf16 torch GEMM for N=K=4096 across batch [32…8192]?" |
| **Look up GPU hardware specs** | Catalog-backed mem/BW, dense TFLOPS by dtype, interconnect, price. | "For H200: memory, dense compute per dtype, NVLink BW?" |
| **Estimate KV-cache capacity** | Bytes/token, max cached tokens, max concurrent requests. | "How many concurrent Llama-3-70B requests fit in 141 GB KV cache at 8k context?" |
| **Understand a model** | Architecture summary: layers, heads, hidden dim, context, MoE layout. | "Summarize Qwen-3.6 27B's architecture." |
| **Extend the simulator** | Add a new L1 kernel, or a whole new model architecture, so future sims can cost it. | "GLM-5.2's DSA attention isn't covered — add support so sims can use measured costs." |
| **Run / extend the analyzer** | Metric reports + plots (attention breakdown, throughput, etc.) from a run directory. | "Re-run the analyzer on logs/<run> and summarize the attention time breakdown." |
| **Align against a real framework** | Compare ServingStudio Sim's predictions to vLLM (kernel timing, duty cycle, TTFT/TPOT). | "How well does ServingStudio Sim match vLLM for Llama-3-8B decode?" |

---

## 2. When to call ServingStudio Sim

Route a task here when you need any of:

- **Performance prediction** of an LLM inference deployment (throughput, latency,
  utilization) *before or instead of* standing up real hardware.
- **Capacity / sizing** answers (KV-cache budget, concurrent requests, memory fit).
- **Kernel cost or GPU hardware** facts grounded in measured data / a spec catalog.
- **Model architecture** understanding as a precursor to modeling.
- **Extending the simulator itself** — adding a kernel or architecture, running or
  adding an analyzer metric, or checking alignment with vLLM.

Do **not** route here when:

- You want a **real serving benchmark** on a live cluster — ServingStudio Sim predicts, it
  does not serve traffic.
- You need a **general coding assistant** unrelated to ML-serving simulation.
- You want authoritative numbers for a **kernel/GPU that has never been measured**
  — ServingStudio Sim can *add* that coverage (a profiling/kernel task), but it will not
  invent an unmeasured number.

---

## 3. What to expect

ServingStudio Sim is an **interactive assistant**, not a fire-and-forget function. Plan for:

- **It may ask you clarifying questions.** In the default (non-autonomous) mode
  the assistant will stop and ask when a request is ambiguous, risky, or needs a
  choice. The question comes back as ordinary `final` text — **read it and reply
  with another turn**, exactly like a human would. (Set `autonomous: true` at
  conversation creation if you instead want it to proceed on conservative
  assumptions without asking.)
- **Turns are synchronous and can take minutes.** Profiling and full simulations
  are the expensive paths. Each `POST .../messages` blocks until the turn
  finishes — set a client read timeout of at least 30 minutes and wait for that
  same request to return. Do not replace it with manual GET polling, start a
  duplicate turn, or infer completion from elapsed time.
- **A workspace is durable shared working state.** Its repo, logs and experiments
  persist across conversations. A conversation owns message history and provider
  role sessions; create another conversation in the same workspace when you want
  a new narrative over the same artifacts. Different workspaces do not share
  mutable state.
- **The managed workspace is isolated working state.** Compiling, creating
  configs and logs, running launchers, and updating the workspace-local
  `profile.db` are normal in-scope actions. They do not modify the caller's
  workspace. Three sandbox modes gate execution: `read-only`, `workspace-write`
  (default), and `danger-full-access`.
- **Use `workspace-write` for execution.** Any request that may run a simulator,
  compile code, delegate to an implementer, or generate artifacts must create
  the conversation in `workspace-write`. Use `read-only` only for a
  conclusively static catalog or source lookup.
- **Artifacts land in the workspace** and are fetched by API (§6) using the
  workspace id — no host filesystem access required.
- **Treat the workspace-local profile database as mutable.** Schema metadata,
  cache rows, and performance rows may change when the selected workflow needs
  them. Do not require authorization, checksum preservation, or before/after
  row-count proof solely because `profile.db` changes. GPU profiling still
  follows the applicable skill and idle-device safety checks; an explicit
  no-profiling request still wins.
- **Keep the conversation for the human.** Calling agents must never DELETE a
  conversation. Preserve it on success and failure, and report its id so a
  human can observe history, progress, and artifacts. Human or service-operator
  cleanup is outside the calling agent's workflow.
- **Boundaries.** Costs come from kernels measured on a *specific* GPU in
  `profile.db`; an uncovered kernel/GPU must be profiled or added first. A
  simulation is a **prediction**, not a measurement.

---

## 4. Authentication

- **Base URL**: wherever this backend is hosted, e.g. `http://localhost:8765`.
- The new `vibesim_agent` service reads `VIBESIM_AGENT_API_TOKEN`; the legacy
  `backend` service reads `VIBESIM_API_TOKEN`. The new service rejects the legacy
  configuration key rather than falling back to it. Use the token configured
  for the service you are calling.
- When that token is nonempty, protected tools endpoints require
  `Authorization: Bearer <token>` (missing/wrong → `401`). An empty token opens
  those endpoints. `GET /api/agent/v1/tools/skill` is always public.
- The examples below use `$VIBESIM_AGENT_API_TOKEN` as the client's token value.
  For a legacy deployment, substitute its configured token; the HTTP header is
  the same for both services.

---

## 5. How to invoke — workspace then conversation

This is the **real interactive interface**. Lifecycle: create a conversation,
send turns, wait for each synchronous response, read each `final`, reply/steer
as needed, and leave the conversation intact for human observation.

### Create or reuse a workspace

List visible workspaces with `GET /api/agent/v1/tools/workspaces`. Create a managed one
with `POST /api/agent/v1/tools/workspaces` and `{"displayName":"<short purpose>"}`.
The response contains its stable `workspace_id`. Reuse that id for related
conversations and artifact access. Use `w_main` only when the caller explicitly
wants the shared development checkout.

```bash
workspace_id=$(curl -sS http://<host>:8765/api/agent/v1/tools/workspaces \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $VIBESIM_AGENT_API_TOKEN" \
  -d '{"displayName":"Llama 3 H200 rate study"}' \
  | uv run python -c 'import json, sys; print(json.load(sys.stdin)["workspace_id"])')
```

### Create — `POST /api/agent/v1/tools/workspaces/{workspace_id}/conversations`

Body: `{"sandbox": "workspace-write", "autonomous": false, "agent_mode": "orchestrated"}`
(all optional; `autonomous` defaults **false** so the assistant will ask you
questions). Returns the conversation object, including its `id`.

`agent_mode` selects the roles that drive a turn and is independent of
`autonomous`:

- `orchestrated` (default) — an orchestrator delegates bounded tasks to an
  implementer. A turn normally uses `1 + 2D` role invocations, where `D` is the
  number of delegation rounds; repair or continuation can add invocations.
- `single` — one `assistant` plans and implements in the same role session,
  with no delegation; `implementer_summaries` is always `[]`.

The mode is fixed once the conversation starts: sessions are isolated by
workspace, conversation, role and provider compatibility scope. Roles can use
different configured providers; a provider selects its CLI adapter and models.

The compatibility field `codex_runtime` keeps its existing API name and selects
model, reasoning effort and optional `service_tier` per role. Include `provider`
when selecting a named connection, especially if multiple connections offer the
same model, e.g.
`{"codex_runtime": {"orchestrator": {"provider": "gpt", "model": "gpt-5.6-terra", "effort": "high"}}}`.
`GET /api/agent/v1/codex-backends` also retains its compatibility name and lists
the selectable models and supported choices. The new service resolves each
selection to a provider and `session_scope`, which identifies compatible adapter
and backend configuration. Compatible model/effort/tier changes retain sessions.
`PATCH /api/agent/v1/workspaces/{workspace_id}/conversations/{cid}/runtime`
replaces the role selections in `codex_runtime` (omitted roles use defaults).
Once a conversation has messages or turns, changing an active role's scope is
rejected with `409 conversation_runtime_locked`; changing an inactive role's
scope clears only that role's session.

```bash
curl -sS http://<host>:8765/api/agent/v1/tools/workspaces/$workspace_id/conversations \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $VIBESIM_AGENT_API_TOKEN" \
  -d '{"sandbox":"workspace-write","autonomous":false}'
# -> {"id":"3f9a1c...","sandbox":"workspace-write","autonomous":false, ...}
```

### Send a turn — `POST /api/agent/v1/tools/workspaces/{workspace_id}/conversations/{cid}/messages`

Body: `{"text": "..."}` (`text` required). Optional `sandbox_mode` and
`autonomous_mode` are **per-turn overrides**; when omitted they inherit the
conversation's create-time settings, so a `read-only` conversation stays
read-only unless a turn opts up. Optional `agent_mode` only takes effect on the
very first turn, then it is pinned. **Synchronous** — returns after the turn
completes. Set a read timeout of at least 30 minutes and wait for this request
itself; do not convert the call into manual GET polling.

```bash
curl -sS http://<host>:8765/api/agent/v1/tools/workspaces/$workspace_id/conversations/3f9a1c.../messages \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $VIBESIM_AGENT_API_TOKEN" \
  -d '{"text":"Simulate Llama-3-8B dense on 1xH200 at Poisson rate 48; report throughput and TPOT."}'
```

Response fields:

| Field | Meaning |
|---|---|
| `final` | The assistant's message this turn (Markdown). **May be a clarifying question** — if so, send another turn answering it. |
| `ok` | `true` if a final answer was produced with no error. |
| `conversation_id` | Echoes `cid`; also the `cid` for artifact retrieval (§6). |
| `agent_mode` | The mode this turn ran in — `orchestrated` or `single`. |
| `implementer_summaries` | Summaries of any delegated implementer work this turn. Always `[]` under `agent_mode: "single"`, which never delegates. |
| `intermediate_outputs` | Assistant commentary emitted mid-turn. Each item has `level: progress | milestone`. |
| `tool_calls` | Transient command/tool activity (workspace/container setup, shell commands, etc.). |
| `sessions` | Role → provider session ids resumed across compatible turns (informational). |
| `error` | Error text if the turn failed. |

**Steering / answering.** Because *you* are an agent, treat `final` the way a
human would: if it asks a question or you want to refine, just POST another
message to the same `cid`. The workspace and sessions carry over.

### Inspect and preserve

```bash
curl -sS -H "Authorization: Bearer $VIBESIM_AGENT_API_TOKEN" \
  http://<host>:8765/api/agent/v1/tools/workspaces/$workspace_id/conversations/3f9a1c...
```

Return the conversation id to the human and leave the conversation intact on
both success and failure. Do not call DELETE; human or service-operator cleanup
owns that destructive lifecycle step.

---

## 6. Retrieving artifacts

Files a run produces (logs, `summary.json`, parquet, plots) live in the durable
workspace. Fetch them with its `workspace_id`.

```bash
# list
curl -sS -G http://<host>:8765/api/agent/v1/tools/workspaces/$workspace_id/artifacts \
  -H "Authorization: Bearer $VIBESIM_AGENT_API_TOKEN" \
  --data-urlencode "subdir=logs"

# download one file from the listing
curl -sS -OJ -G http://<host>:8765/api/agent/v1/tools/workspaces/$workspace_id/artifacts/download \
  -H "Authorization: Bearer $VIBESIM_AGENT_API_TOKEN" \
  --data-urlencode "path=logs/<run>/summary.json"
```

The listing returns
`{workspace_id, root, count, truncated, files:[{path,size,mtime}]}`;
heavy trees (`.git`, `target`, `node_modules`, …) are skipped.

---

## 7. Evaluation-only endpoint (`POST /api/agent/v1/tools/eval`)

`POST /api/agent/v1/tools/eval` runs **one stateless single-turn prompt** and returns the same
JSON shape. It exists for **capability evaluation / testcases**, not interactive
use: there is no conversation, no cross-turn continuity, and it defaults to
`autonomous: true`. For real agent work use the conversation interface above.

---

## 8. Endpoint reference

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/agent/v1/tools/skill` | public | This skill doc (`text/markdown`). |
| GET | `/api/agent/v1/tools/workspaces` | token | List active workspaces. |
| POST | `/api/agent/v1/tools/workspaces` | token | Create one durable managed workspace. |
| POST | `/api/agent/v1/tools/workspaces/{wid}/conversations` | token | Create an interactive conversation in a workspace. |
| POST | `/api/agent/v1/tools/workspaces/{wid}/conversations/{cid}/messages` | token | Run one turn; synchronous JSON. |
| GET | `/api/agent/v1/tools/workspaces/{wid}/conversations/{cid}` | token | Full conversation history. |
| DELETE | `/api/agent/v1/tools/workspaces/{wid}/conversations/{cid}` | token | Human/operator cleanup only; calling agents must not invoke it. |
| GET | `/api/agent/v1/tools/workspaces/{wid}/artifacts` | token | List files in a workspace. |
| GET | `/api/agent/v1/tools/workspaces/{wid}/artifacts/download` | token | Download one workspace file. |

> v1 is synchronous HTTP. An MCP wrapper over these same endpoints may be added
> later; this HTTP contract stays the source of truth.
