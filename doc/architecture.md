# Agent Architecture

Agent owns durable workspace, conversation, turn and managed-job lifecycle.
ServingStudio UI owns browser presentation. Rust Analyzer owns numerical results and
resource catalogs; Agent links to those results by stable resource identity.

## Component Boundaries

| Component | Responsibility |
| --- | --- |
| `bootstrap.py`, `settings.py` | Explicit configuration and dependency assembly |
| `api/` | HTTP validation, authentication, projection and SSE transport |
| `services/` | Conversation, turn, workspace, naming and managed-job lifecycle |
| `domain/` | Roles, requests, decisions, events, outcomes and evidence contracts |
| `providers/` | Named connections, capabilities, CLI adapters and output translation |
| `runtime/` | Docker ownership, mounts, provider homes and subprocess cleanup |
| `storage/` | SQLite transactions, workspace descriptors, registry and ownership |
| `prompts/` | Role contracts and deterministic prompt rendering |
| `tools/` | Offline conversion, migration selection and deployment auditing |

Imports do not initialize state or start Docker. `init` creates an absent state
root; `serve` validates initialized state, takes its exclusive ownership lock and
recovers interrupted turns before accepting requests. The service uses one process.
Shutdown drains active work before releasing state ownership.

## Workspaces And Conversations

A managed workspace owns one copied repository and its logs. Its conversations
share those files and serialize turn execution within that workspace. `w_main`
refers to the external development checkout. Editing or committing that checkout
is an ordinary user workflow, not a reason to require legacy migration again.

Each conversation has isolated role/provider runtime homes and one owned Docker
container. Removing the container preserves durable state. Conversation deletion
waits for its turn and removes conversation-owned state without deleting the
workspace repository or result artifacts.

`orchestrated` mode runs an orchestrator and an implementer. The orchestrator
returns decisions, can delegate work and resumes after the implementer handoff.
`single` mode runs one assistant. Mode and autonomous policy become fixed once
history exists. Supported sandbox modes retain the prompt and delegation policy;
the shared workspace bind is writable, so read-only mode is not an OS sandbox.

## Turns And Streams

Browser and tools requests share one durable turn service. Disconnecting a stream
does not cancel its turn. Events are persisted in sequence; reconnecting reads
that durable stream. Historical messages retain stable IDs, nullable legacy turn
IDs and original metadata. Read projections handle legacy event shapes without
rewriting historical rows.

Cancellation targets a specific turn. Readiness tracks when a role has executed
enough to preserve a useful resume role; a timeout bounds deferred interruption.
Provider adapters own remote invocation signals, pipe cleanup and idle deadlines.
Restart recovery interrupts unfinished turns, removes their owned containers and
retains messages and compatible provider sessions.

The driving role has two decision-repair attempts and three checkpoint
continuations. Exhaustion is a failed turn, not a successful final answer.
Completed implementer work remains in history when a later call fails.

## Providers And Runtime

A named provider identifies one CLI connection. A role selection fixes provider,
model, effort, service tier and session scope. Scope binds adapter, connection ID,
endpoint and applicable profile home; model, effort and token rotation do not
change it. Incompatible populated conversations reject runtime changes. Prompt
fingerprints record provenance and do not discard sessions.

The runtime copies selected credentials/configuration into isolated homes without
copying host conversation history. Provider-specific environment variables are
supplied only to that adapter. See [Provider Connections](providers.md).

Runner images contain dependencies and a Cargo target seed. Container preparation
checks image identity and environment, mounts the selected prompt and seeds an
absent workspace target. It does not compile first-party binaries. The launcher
builds source-dependent binaries when required. Submodules, role instructions,
MCP code, optional peer repositories and model caches use read-only bindings.

## Integrations

Browser routes and tools routes share services. The tools API token does not
authenticate every browser route. Managed callbacks instead require a capability
scoped to a running turn; legacy callback aliases remain available for existing
workspace copies. See [Managed Jobs](managed-jobs.md).

Analyzer citations are frozen against their turn's dictionary. Agent must not
reconstruct numerical results from messages or managed-job rows. Agent and Analyzer
must use the same active workspace registry.

Citation registration accepts current `/api/analyzer/v1/` subject paths and
legacy `/api/v1/` paths. Its response retains the cumulative dictionary and adds
`registeredEntries` for the resource read by this request. MCP uses those entries
to attach citations to the returned result; persisted turn events keep the
cumulative dictionary for final-answer freezing and replay.

Legacy conversion is a separate operation into an independent target. It preserves
historical data and reports validation; provider resume requires an additional
real CLI check. Migration controls are cutover artifacts, not permanent normal
startup dependencies. See [Migration](migration-v1.md) and [Cutover](cutover.md).
