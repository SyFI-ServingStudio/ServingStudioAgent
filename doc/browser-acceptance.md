# Browser Acceptance Before Legacy Frontend Retirement

This record compares the retained Agent frontend with the integrated VibeSimUI.
The capability matrix is the original baseline inventory; the completed sections
below contain subsequent live evidence and supersede its "missing" column.
This is not permission to remove the old frontend before deployment acceptance.

The UI inspected is `VibeSimUI` main at `b588138`. Its untracked
`new-design.md` was not used or changed. Agent paths below refer to this refactor
worktree; UI paths refer to `../../VibeSimUI/app/` from this document.

## Evidence Levels

- Unit/component tests establish local behavior with synthetic state or mocked
  transport. Their existence is not evidence of a live Agent connection.
- `e2e/next-chat.spec.ts` intercepts catalog, workspace, conversation, message,
  stream and cancel APIs with `page.route`. Its browser interactions exercise
  production UI components against simulated responses.
- `e2e/file-preview.spec.ts` intercepts file/meta and Analyzer catalogs with
  `page.route`; it does not read a real Agent workspace.
- Prior private Agent HTTP/SQLite, actual Uvicorn stream/cancel/reconnect,
  provider and Analyzer rehearsals validate server behavior. They did not drive
  this UI through a browser against the new Agent and cannot complete this table.

## Capability Matrix

Paths in the implementation/test columns are relative to `VibeSimUI/app`.
"Missing" means no matching live new-Agent browser evidence was identified in
this review, not that the implementation is absent or a test was run and failed.

| Capability and retained frontend reference | Current UI implementation | Existing local or mocked coverage | Missing real new-Agent browser evidence |
| --- | --- | --- | --- |
| Sandbox/autonomous/agent_mode settings: `frontend/src/main.tsx` selection, localStorage preferences and restoration from stored conversation | `src/app/AgentHost.tsx`, `panels/conversation/ConversationSurface.tsx`, `session/draft.ts`. Mode/autonomous are configurable; `draft.ts` currently hardcodes `sandbox: 'workspace-write'`, so the old sandbox selector is not fully carried over | `AgentHost.test.tsx` and `ChatPage.test.tsx` assert creation and first-message settings with mocked transport | Confirm whether fixed workspace-write is an accepted product change or restore the sandbox choice before claiming parity. Exercise mode/autonomous preferences and restored conversation settings against real create/history responses. Check draft preference restoration separately from server-owned settings and per-turn overrides. |
| Human conversation deletion and sidebar refresh: `frontend/src/main.tsx` delete action | `src/app/AgentHost.tsx` history deletion, `session/api.ts` (`deleteConversation`) | `session/api.test.ts` checks the canonical DELETE request with mocked fetch | Delete only a designated synthetic conversation from the browser; verify refreshed list removal, unavailable deep link and server cleanup, without changing sibling conversations. Calling agents' separate rule to preserve conversations remains unchanged. |
| History pagination and conversation deep links: `frontend/src/main.tsx` (`loadOlderMessages`), `conversationLocator.ts`, `api.ts` | `src/session/controller.ts` (`loadEarlier`), `session/api.ts` (`limit`/`before`), `app/agentLocation.ts`, `panels/conversation/ConversationHistory.tsx`, `ChatPage.tsx` | `session/controller.test.ts`, `session/api.test.ts`, `app/agentLocation.test.ts`; `e2e/next-chat.spec.ts` opens an address with empty browser storage and exercises history navigation, using routed responses | Seed more than one actual history page with stable IDs/turn IDs; open a canonical URL in a fresh browser, load older messages without duplicates, reload and use back/forward across two conversations. Check persisted ordering and workspace identity against SQLite. |
| Draft first message exactly once: `frontend/src/main.tsx` conversation creation/send flow | `src/session/draft.ts`, `app/AgentHost.tsx`, `panels/conversation/ChatPage.tsx` | `app/AgentHost.test.tsx` checks draft replacement after canonical turn start; `ChatPage.test.tsx` covers late creation, leaving/closing a draft and selected settings; mocked `next-chat.spec.ts` covers draft replacement without extra history entries | Count real conversation creation and message POSTs while submitting a draft and navigating/closing during creation. Confirm one durable user message and one turn, with no second POST when the new URL mounts. |
| Full role runtime selection and locks: `frontend/src/main.tsx` runtime picker, `types.ts`, `api.ts` | `src/panels/conversation/CodexRuntimePicker.tsx`, `codexRuntime.ts`, `session/api.ts` (`updateConversationRuntime`) | `CodexRuntimePicker.test.tsx` covers per-role model/effort/tier, unavailable Claude, tier reset, single mode and family locking; API tests cover the request boundary. New Agent `tests/test_runtime_patch_v2.py` separately covers scope locks and transaction behavior | Against actual catalog and PATCH routes, verify all three role fields survive edits, omitted-role defaults follow full replacement, same-scope edits retain sessions, and a server `409 conversation_runtime_locked` is shown without optimistic state corruption. UI family-based availability must not be mistaken for authoritative server `session_scope` validation. No real provider is needed merely to inspect seeded locks. |
| SSE attachment/reconnection; closing chat does not Stop: `frontend/src/api.ts`, `main.tsx`, `turn.ts` | `src/session/controller.ts`, `stream.ts`, `useSession.ts`; host lease ownership in `app/AgentHost.tsx` | `controller.test.ts` covers lost streams, reattachment, late history and released leases; `stream.test.ts` covers framing. Mocked `next-chat.spec.ts` explicitly checks closing a running chat without cancel | Use real incremental TCP SSE from new Agent. Close/unmount and reopen the panel while a bounded turn remains running; verify no cancel request, no duplicate turn POST, and one persisted final after reconnect. Existing direct-HTTP reconnect evidence is useful but does not exercise browser lease behavior. |
| Targeted Stop and repeated cancellation: retained `frontend/src/main.tsx` Stop action and `api.ts` | `src/session/controller.ts` tracks the POST/GET `X-Turn-Id`; `session/api.ts` sends the target; `ConversationSurface.tsx` presents state | `controller.test.ts` includes naming the original and reattached turn, lost POST responses, cancellation races and not stopping another turn; `api.test.ts` and mocked `next-chat.spec.ts` cover requests/UI | Observe the browser request's exact turn ID, interrupt a bounded running tool, and confirm durable interrupted state. Repeated or late Stop must not cancel a subsequent turn. Verify UI settles after final history and compare with the real server's cancellation evidence. |
| Markdown images, workspace links and artifact access: `frontend/src/markdown.ts`, `components/Turn.tsx` | `src/panels/shared/MarkdownBody.tsx`, `session/workspaceFiles.ts`, `panels/file/FilePreviewPage.tsx`, `ConversationTranscript.tsx` | `MarkdownBody.test.tsx`; `e2e/file-preview.spec.ts` checks syntax, search, terminal escape rendering and canonical navigation with mocked file responses | Serve a synthetic PNG, text and terminal log from a real private workspace. Follow Markdown image/file links and artifact download; verify actual bytes, workspace ID, URL encoding, missing/denied path handling and browser rendering. A mocked image/file URL is insufficient. |
| Five citation destinations: retained `frontend/src/markdown.ts` and `components/Turn.tsx` render result-bearing assistant content; full canonical result navigation belongs to integrated UI | `src/session/citation.ts`, `evidenceRef.ts`, `app/citationDictionary.ts`, `evidenceLocation.ts`, `evidence.ts`, `panels/shared/MarkdownBody.tsx` | `app/evidenceLocation.test.ts`, `evidence.test.ts`, `MarkdownBody.test.tsx`, `session/analyzerContext.test.ts`: run, sweep, prediction, kernel profile and kernel measurement decoding/navigation, malformed/unsupported cases | Seed actual new-Agent history with frozen dictionaries and citations for all five types, backed by a private matching Analyzer catalog. Click each token and verify workspace/resource/panel/cursor restoration, including docked chat. Confirm broken refs remain inert and changing current Analyzer context does not rewrite old citations. |
| Managed jobs and result jumps: retained `frontend/src/turn.ts`, `components/Turn.tsx` job/activity presentation | `src/session/projection.ts`, `panels/conversation/ConversationTranscript.tsx` job cards, `app/evidence.ts` (`managedResultLocation`), `app/App.tsx` navigation | `session/projection.test.ts`, `panels/conversation/sessionView.test.tsx`, `app/evidence.test.ts` cover card/result mapping and reject unknown kinds/invalid IDs. Agent's separate managed protocol tests cover server registration | Produce or seed simulation and three typed job records/events through the actual new Agent, with matching private Analyzer resources. Verify running/completed/cancelled cards and click destinations use workspace/resource IDs rather than filesystem guesses. Test unavailable artifacts explicitly. |

## Bounded Next Acceptance

### Completed Live Subset

A private run now connects the real VibeSimUI through Vite to the new Agent HTTP
application and a matching Rust Analyzer. It uses real SQLite, routes and turn
lifecycle, with a controlled provider adapter and Docker subprocess substitute.
No Agent or Analyzer HTTP response is intercepted. The final local evidence is
`tmp/agent-ui-live-r3we2gfe/{report,browser}.json` at the workspace root; the report
binds Agent sources, UI diff, Analyzer binary and harness hashes, and confirms
source stability during the run.

The run verifies all 120 seeded history messages in order after pagination and
lazy mounting, restored sandbox, exactly one draft creation/message/turn,
disconnect and reconnect without another message or cancellation, targeted Stop
using the original response turn ID, and a single persisted cancelled answer.
It checks the complete original message rows remain unchanged, real text preview,
embedded PNG decoding and PNG bytes, and 1440px/390px screenshots. Test services
exited normally and their three ports were confirmed free. Independent review
checked the logs, database and final mobile screenshot.

This exposed and fixed a UI settings restoration bug: repeated effects on a
cached conversation could restore defaults over historical settings. The new
StrictMode regression fails before the fix, passes afterward, and preserves an
unsent local sandbox choice across history refresh. A container-width layout
rule also keeps long runtime labels from clipping the autonomy tag; actual
visual evidence here covers the full-page desktop and mobile views, not every
docked width.

This transport subset does not complete the matrix. The result-navigation subset
below adds citation and ready-job evidence. The runtime/deletion subset below
adds those workflows. The history subset below adds pagination anchors and a late
page response across conversation navigation. Other job states and additional
citation selectors remain. These subsets do not establish real
provider/Docker execution or production migration acceptance.

### Completed Result Navigation Subset

The final report is `tmp/agent-ui-results-pszzf92l/{report,browser}.json` at the
workspace root. It drives the real UI, new Agent HTTP application and matching
Rust Analyzer without HTTP interception or provider requests. Synthetic artifacts
include a valid run summary/concurrency payload, a one-member sweep, a timing
prediction with a real one-row Parquet cost log and one cost-tree leaf, a profile
curve, and a measurement summary with valid synthetic PNGs. They are navigation
fixtures, not performance measurements.

The Agent's actual dictionary builders and `freeze_citations` generate the five
historical citations. The actual job service registers simulation, timing_predict,
kernel_profile and kernel_measure jobs and advances them to ready; production
history projection generates their cards. Browser checks cover all nine clicks,
exact workspace/resource identity, relevant panel/metric/coordinate selectors,
the retained docked conversation, real target payload responses and visible
result content. Each history reload rechecks the original citation text and
offsets. The checks wait for network idle and reject visible parse/load errors.

Every SQLite table is compared in full before and after navigation: 122 messages,
two conversations, four jobs, one experiment, one turn, 12 events and their other
storage relationships remain unchanged. The report binds source/harness and
artifact hashes, including the private model configuration. Owned processes exit
normally. Desktop screenshots include all nine destinations; a 390px history
screenshot and DOM bounds checks verify long citation IDs fit both mobile and
the narrow desktop dock. A three-line citation-button wrapping fix addresses the
clipping found during this run; type checking, 16 focused tests and lint pass.

The run fixture intentionally omits nine non-target resources. Only their exact
run-ID-bound endpoints returning `404` with `code=artifact_missing` are accepted
and listed under `expected_missing_resources`; any target error, other HTTP error
or visible parse error fails. Screenshots preserve those not-generated states.
This does not claim a fully populated run analysis, numerical accuracy, other job
states, every citation selector, or missing-reference behavior. Earlier failed
reports remain available, including `xx9wo8tg`, whose HTTP checks passed but whose
run-summary fixture failed visual inspection; it is not final visual evidence.

### Completed Runtime And Deletion Subset

`tmp/agent-ui-workflows-2fkaykkg/{report,browser}.json` records real UI/Agent HTTP
and SQLite checks with a matching private Analyzer. The provider adapter and Docker
remain controlled substitutes; no HTTP response is intercepted.

The UI changes orchestrator and implementer model/effort/tier, submits the full
three-role runtime PATCH, then posts exactly one message. All three saved sessions
survive unchanged; the controlled orchestrator reuses its original session and
finishes one turn. This does not establish an implementer provider resume.

A historical single-role conversation has an old scope under the same catalog
family. Its PATCH returns `409 conversation_runtime_locked`, displays an error,
retains the draft and issues no message POST. All associated SQL rows, including
timestamps and sessions, remain unchanged. The UI cancels non-success response
bodies, so a transparent ASGI send observer records the same actual response body
for the exact-code assertion; the browser checks the status and visible behavior.
The first run `9lnpga5g` failed when Playwright tried to reread that cancelled body.

Two-step deletion removes only the designated conversation, refreshes history,
navigates away and makes its history endpoint return 404. Before/after SQL checks
verify conversation-owned rows and runtime directory disappear while the experiment,
repository/log bytes and protected 120-message history remain unchanged. Source,
harness and Analyzer hashes stay stable; all owned processes and ports are released.
This covers completed-conversation deletion, not deletion during a running turn,
and the 404 assertion is an API check rather than a deleted-page UI rendering test.

### Completed History Navigation Subset

`tmp/agent-ui-history-n8oyc8s3/{report,browser}.json` records desktop Chromium
(1440x900) using real UI/Agent HTTP, SQLite and a matching private Analyzer, with
no provider call, HTTP interception or mutating request. All database rows and
private repository bytes remain unchanged, source/harness hashes are stable, and
owned processes, ports and tmux sessions are released. Independent review checked
the database snapshots, request order, screenshots and final native browser script.

The browser loads exactly 50, 100 and 120 ordered message blocks. Returning to the
pagination button uses real wheel input. Each old-message anchor is visible and
unchanged at click time; six successive pairs of animation frames after each
prepend stay within 3px. Maximum measured displacements are 0.40625px and 0.078125px.
A single bounded server gate delays an actual older-page request until the second
conversation has loaded. Releasing that response leaves the second conversation's
text, runtime and sandbox unchanged. Browser back/forward restores the correct
conversation identity and message order.

This exposed two UI problems. Loading-state renders consumed the pending scroll
restore before the older page arrived, and deferred content could shrink above
the reader after the initial restore. Restoration now waits for the preceding
message boundary, uses the visible message rather than total height, and observes
transcript resizing while that reading position remains active. Wheel, touch,
pointer or keyboard input, another page request, an outline jump, sending and
unmounting cancel preservation. Cancellation starts while the page is still in
flight, so a delayed response cannot override a newer user action.

Type checking, 95 related tests and scoped lint pass. Regressions distinguish
height changes above/below the reader, late resizing, user input while waiting,
observer cleanup and detached anchors. Earlier failed reports remain available;
`48n2p0qm` contains the diagnostic trace proving the post-restore 24px shrink.
The accepted run has no scroll-property overrides. It is not evidence for every
browser, arbitrary layout reordering, mobile pagination or every navigation race.

### Remaining Acceptance

The recorded runs cover the retained frontend's principal history, settings,
conversation lifecycle, streaming, cancellation and workspace-file workflows,
plus integrated UI citation and ready-job navigation. The companion fixes remain
unmerged and undeployed. Each completed section states its exact scope; untested
cross-products of kinds, states, browsers and viewports are not additional
retirement requirements without a concrete uncovered behavior or regression.

The supplementary run `tmp/agent-ui-status-97d_gvj_/{report,browser}.json` passed
independent review. Eight real JobService registrations/updates and production
history projection yield three running, four stopped and one ready card; all
seven non-ready cards are disabled. A frozen citation naming an absent prediction
stays on the chat route with an unavailable indication, while an unregistered
literal remains plain code. The ready card opens that prediction's exact missing
result page with the workspace and docked conversation retained. Only its exact
descriptor/cases endpoints may return 404; an independently labelled HTTP probe
confirms `code=prediction_not_found` rather than claiming to inspect the browser's
same response body. All database rows and repository bytes remain unchanged,
source/harness/Analyzer hashes are stable, and owned processes/ports are cleared.
Screenshots and final evidence were independently reviewed. This closes shared
historical negative-path presentation; it does not prove active job termination,
model execution, or every state/type combination. The job protocol uses
`interrupted`; `cancelled` is a turn outcome.

The remaining deployment conditions are the original plan's Claude historical
dual-role session migration/resume verification and the actual Phase 6 procedure:
freeze component revisions and runtime configuration, stop owned old writers,
make the final consistent backup, migrate and validate the selected state root,
switch the matching Agent/Analyzer/UI deployment, smoke-test before reopening
writes, observe, and retire legacy source/entry points with rollback artifacts
retained. See `cutover.md` and `plan.md`. Existing private GPT and migration
rehearsals do not substitute for the missing Claude or production steps.
