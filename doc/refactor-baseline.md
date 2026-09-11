# Agent Architecture Baseline

The implementation starts from Agent `e26ad6d`. `plan.md` defines the target;
this file distinguishes observed behavior from requirements and known defects.

## Evidence

- `tests/test_http_contract.py` exercises the ASGI application with real turn
  orchestration and SQLite, replacing only CLI, Docker, and workspace provisioning.
- Existing CLI parsing, subprocess, rollout, cancellation, store, citation and
  workspace tests remain necessary. HTTP tests do not replace them.
- `tools/snapshot_v1.py` inventories decorated API routes and their direct
  dependencies without importing the application. It also records per-database
  schemas, row counts, content digests, foreign-key and undeclared-link errors using read-only
  transactions. It includes descriptors omitted from the active registry.
  After retiring `backend/`, run it with `--repo` pointing to the retained
  legacy checkout. Missing legacy source or an empty route inventory is rejected;
  this tool does not inventory the new router-based application.
- The inventory is not a migration backup: databases are captured individually,
  and active runtime/session files are not frozen. Full stopped-state migration,
  file inventory and real provider resume verification remain pending.
  WAL reads can update SQLite shared-memory coordination; read-only here means
  no application-data writes. Dirty peer diff hashes exclude untracked contents,
  so they identify an observation, not a fully frozen integration baseline.

## Wire Behavior

Browser endpoints live under `/api/agent/v1`, tools under `/api/agent/v1/tools`.
Browser reads/writes are public. Tools require the configured API token, or are
open when it is unset; the skill document is always public. Internal callbacks
require a separate scoped capability, not the API token.

Live POST and reconnect GET identify the turn with `X-Turn-Id`. Reconnect starts
from the beginning of the retained live stream; idle reconnect returns 204.
Disconnecting a browser leaves its turn running. A targeted Stop for a different
turn returns `cancelled: false, stale: true`; repeated Stops must not interrupt
cleanup a second time. The interrupted role is stored before another message
can start. Existing failure and cancellation text/outcomes remain distinct.

Messages expose stable integer `id` and nullable `turn_id`; old rows keep null.
Prompt fingerprint changes preserve sessions. Compatible model changes preserve
sessions, while incompatible active-role families remain locked after starting.
Frozen analyzer citations keep their v2 bytes, including dynamically registered
dictionaries; the final answer uses the latest dictionary from its own turn.

## Known Differences To Resolve

1. Historical replay contains original persisted events; live SSE transforms
   events and finishes with `done`, which is not persisted. They are not identical
   wire streams. Keep old stored bytes; define historical normalization at the
   reader boundary rather than rewriting the database.
2. The synchronous API has its own lock and cleanup, lacks the browser's active
   turn registration, and can strand a running record on cancellation. The shared
   service must fix this, not reproduce the bug as a compatibility requirement.
3. Existing launchers still use `/api/internal/managed-{runs,jobs}`. The baseline
   removed these addresses; the replacement must restore thin aliases until all
   managed workspace copies migrate. An image rebuild does not update those copies.
4. The new evidence MCP expects redesigned Analyzer paths and resource grammar.
   UI and Analyzer worktrees are still changing; final integration baselines must
   be pinned before their end-to-end acceptance can be declared complete.
5. The old driver labels exhausted decision repairs or checkpoint continuations
   as `final_answer` even though no valid terminal decision was produced. The new
   single driver reports these as failed turns, preserving the two-repair and
   three-continuation budgets. API projections must preserve this semantic fix;
   this outcome is intentionally not byte-compatible with the old defect.
6. Conversation creation now rejects sandbox strings outside the three supported
   modes with HTTP 422. The old handler could persist arbitrary strings; retaining
   those invalid inputs would bypass the typed domain boundary. All supported
   sandbox values retain their existing meaning.
7. Claude no longer emits `role_ready` merely because a failed process finishes,
   nor for empty or malformed messages. A top-level result or actual assistant
   text/tool block establishes readiness. This prevents cancellation from saving
   a resume role without evidence that the role executed.
8. New bind mounts use explicit `--mount` fields and reject commas, double quotes
   and control characters in paths, including resolved symlink sources. Spaces,
   colons and equals signs remain supported. Host paths must be explicitly
   supplied as absolute paths; composition owns user-home expansion. The existing
   workspace `AGENTS.md` target must be a regular file, not a symlink or directory.
   These checks prevent ambiguous Docker field parsing and target creation or
   symlink traversal in a shared workspace. Missing peer directories still skip
   the optional mount; a configured missing HF cache still rejects preparation.
   Actual Docker enforcement of nested read-only mounts remains a runtime gate.

## Provider Session Migration

Built-in session scopes have the form `<adapter>:<provider>:v1:<sha256>`.
Codex scopes bind the canonical profile home and effective TOML `model_provider`,
`base_url`, `wire_api` and `requires_openai_auth`; a selected TOML profile is
resolved before those fields are read. Claude scopes bind its explicit
`ANTHROPIC_BASE_URL`. Adapter and provider identities are always included.
Model, effort, service tier, credential values and unrelated CLI settings do not
change the scope. Endpoint URLs with credentials, query strings or fragments are
rejected rather than treating secret material as a session identity.

Legacy family `gpt` (backend alias `traditional`) maps to the registered `gpt`
provider's current Codex scope; `deepseek` (`codexds`) maps to `deepseek`'s Codex
scope; `claude` maps to its Claude scope. Model aliases `sonnet` and `opus` map to
the pinned Claude 5 models only when registered. These mappings identify possible
migration destinations, not proof that an old session can resume. Offline
migration must verify the old profile/backend and move or copy role state from
the old `codex/<conversation>` layout to the matching new role/scope home before
preserving a session ID. No runtime import or registry read performs that move.

The supplied profile preparation guard checks backend scope before and after
copying. A change requires registry reconstruction and explicit session handling.
Concurrent edits to host profile configuration are unsupported: these checks do
not make a multi-file profile copy transactional. A real old-session resume test
remains required before migration is accepted.

## Completion Tracking

Phase 0 remains open until route/auth, SSE/terminal, storage and active peer
contracts are captured and the behavioral suite passes on the baseline.
Real provider/container and historical resume checks are separate release gates.
