# Browser Integration Validation

Run these checks against VibeSimUI, Agent and a compatible Rust Analyzer using an
isolated workspace. This document defines validation scope, not a record of a
particular deployment's results. Record exact revisions, configuration, commands
and observed outcomes separately.

## Evidence Boundaries

Component tests cover local state and rendering. Real HTTP checks cover routing,
SQLite and service interaction. Browser checks cover actual navigation, layout,
streaming and error presentation. Controlled provider adapters do not establish
CLI execution; synthetic Analyzer fixtures do not establish numerical accuracy.
State each boundary when reporting results.

Use desktop and narrow mobile viewports. Check console errors, failed requests,
text overflow and visible content. Avoid intercepting successful HTTP responses
when claiming integration coverage. Capture database and workspace inventories
before and after read-only scenarios to detect unintended writes.

## Workflow Checks

| Workflow | Expected behavior |
| --- | --- |
| Workspace lifecycle | Create, rename, archive and restore update Agent and Analyzer discovery consistently |
| Conversation lifecycle | Create both modes; rename; delete only the selected conversation and its runtime state |
| Runtime settings | Select provider, model, effort and tier per role; save the selected connection even when model IDs overlap |
| Compatibility rejection | A populated conversation rejects incompatible scopes; the draft remains and no message is sent |
| History | Load successive older pages with stable IDs, order and reading position; do not mutate stored history |
| Navigation races | A late older-page response cannot alter a newly selected conversation or its runtime settings |
| Streaming | Incremental activity, usage, handoffs and terminal outcome render under the correct turn and role |
| Reconnection | Disconnecting leaves the turn running; reconnect replays without submitting another message |
| Stop and continue | Target the active turn, preserve the interrupted role, and continue using the compatible session |
| Workspace files | File listing, metadata and content resolve within the selected workspace |

For runtime changes, inspect the actual PATCH and subsequent message request.
Check both UI error presentation and the persisted state after rejected changes.
For history pagination, use real wheel/touch input and test layout changes above
the reading anchor as well as a request that finishes after navigation.

## Result Navigation

Exercise frozen citations and ready-job cards for simulation, timing prediction,
kernel profiling and kernel measurement. Verify exact workspace/resource identity,
panel or metric selection, the retained conversation and visible target content.
Reload history and confirm citation text, offsets and dictionaries remain intact.

Running, failed and interrupted jobs must not appear as ready results. Missing
references should show an unavailable state without navigating to an unrelated
resource. An unregistered citation-like literal remains ordinary text. For a
fixture with intentionally missing resources, enumerate the exact expected 404
endpoints; unrelated request failures are test failures.

## Execution Checks

In a separate isolated real-provider test, verify that the CLI receives the
original session ID, appends to its existing transcript and recalls a baseline
marker. Cover both roles for orchestrated mode, service/container recreation,
targeted cancellation and subsequent continuation. A successful new session is
not resume evidence. Never submit production history merely to test a refactor.

Keep provider credentials out of screenshots, logs and reports. Remove only
test-owned resources after collecting evidence. Preserve failed-run diagnostics
separately and do not count failed or empty reports as passing checks.
