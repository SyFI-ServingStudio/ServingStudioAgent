# Agent Deployment Cutover

This is a preparation record, not authorization to switch a running service.
Provider resume, stopped-state migration rehearsal and final image acceptance
must pass before the Phase 6 cutover in `plan.md`. The old deployment remains active.

## Frozen Candidate

The implemented component candidates are Agent `e90d58b`, integrated launcher `38d46f7`
and UI `ece7120`. These are local commits, not deployed revisions. The local
`tmp/agent-refactor-deployment/release.sh` records full commits and artifact hashes;
later documentation-only commits may advance its Agent pin without changing code.
Its UI script now selects `wt-agent-ui-compat`, containing the accepted settings,
citations and pagination fixes. Historical main workspace paths remain unchanged.

Before managed startup can stop the old deployment, the candidate backend checks
the Agent/UI revisions, tracked changes and unexpected untracked files, the
complete integrated main commit and both dependency lock hashes, Analyzer checksum
and local immutable runner image with its uv-lock label. Each bridge also checks Agent source. The current
main checkout still requires launcher integration; this check rejects it before
migration. Existing copied workspaces retain the tested legacy callback aliases.

The accepted Node22 runner image is
`sha256:9dc036db25b06d1c28d6f7dcaac60987e8441788fdc0c26d9dc95a6dc01475a4`
(tag `vibesim-agent-runner:kanzhu-refactor-v12-node22`). Build evidence is
`tmp/agent-final-image-96v01nld/report.json`; real runtime evidence is
`tmp/agent-runtime-real-fft3c7rt/report.json`. The accepted Analyzer executable is
`tmp/agent-analyzer-build-mwml3zse/repo/target/release/analyze`, SHA256
`1a3ff3f88d1d936fa87443e6e5fc3a283a84b651a6aadc9d2f38d844f56f13aa`.
The candidate scripts pin these artifacts. The older image observations below
describe earlier evidence and the original deployment, not this candidate.

Version checks do not establish provider acceptance or current deployment
ownership. A fresh stop audit, final backup/migration,
production cutover and observation are still required. No candidate script has
been installed or started; only read-only release checks have run.

## Private Acceptance

The final Agent regression passed 828 tests in 40.058 seconds after fixing Docker
management commands to use closed standard input. Provider prompt pipes are unchanged.
The following real checks used the accepted image, private synthetic repositories
and official provider connections; they did not send production history.

- Claude legacy migration: `tmp/agent-claudeme-tracking-final/baseline.json` and
  `migration-resume.json`. The frozen old backend created both role sessions;
  both retained their original IDs, recalled exact identifiers and advanced their
  original transcripts across two new application lifespans and container recreations.
- Mixed delegation: `tmp/agent-mixed-orchestration-kynwx7p7/report.json`.
  Real HTTP ASGI orchestration ran Claude, Codex, then the original Claude session;
  the implementer wrote the specified file and history, SSE and replay agreed.
- Cancel then continue: `tmp/agent-cancel-resume-final/report.json`.
  Real TCP cancellation stopped an observed live tool process before container
  removal; the next turn used the same Codex session and recalled the exact marker.

All three reports passed source/profile preservation and private credential/container
cleanup checks. The TCP test also released its service port. Earlier failed fixtures
remain recorded separately; in particular, the original Claude marker fixture omitted
the implementer role prefix and received a refusal. It is not a passing baseline.
These results complete private provider acceptance, not production migration,
observation, legacy retirement or an actual OAuth refresh.

## Observed Deployment

On 2026-09-11, the workspace's five `agent-workspaces/services/*.sh` scripts
describe this deployment. Script contents are not proof of a process's current
environment; recheck the owned processes at cutover.

| Service | Current Address | Target Change |
| --- | --- | --- |
| Agent | `127.0.0.1:8765` | New package and migrated state root |
| Agent bridge | `172.17.0.1:18765` | Forward to the same Agent port |
| Analyzer | `127.0.0.1:8787` | Read the migrated `registry.json` |
| Analyzer bridge | `172.17.0.1:18787` | Forward to the same Analyzer port |
| UI | `127.0.0.1:5177` | Preserve proxy targets and strict port selection |

These are existing deployment ports, not ports for a concurrent rehearsal.
Check occupancy and ownership before changing services. Use the root
`reproduce.md` per-user convention for a separate deployment and never silently
change ports. Revalidate the Docker bridge address on the actual host.

The five services were rechecked on the dedicated tmux socket
`/tmp/tmux-1003/vibesim-kanzhu` (`tmux -L vibesim-kanzhu`), with one pane per named
service and each pane launching its corresponding script above. This is not a
session named `vibesim-kanzhu` on the default socket. The default socket hosts
other work, and root `scripts/services.sh` uses yet another socket for its three
services. Neither is an interchangeable stop command for this deployment.
The new `tools/tmux_deployment.py` adapter targets only an explicitly audited
dedicated server. Do not treat script names or this observation as a permanent
process identity; capture and validate live evidence before an actual cutover.

## Configuration Changes

| Old Agent Key | New Host Key |
| --- | --- |
| `VIBESIM_WORKSPACES_ROOT` | `VIBESIM_AGENT_WORKSPACES_ROOT` |
| `CODEX_DOCKER_IMAGE` | `VIBESIM_RUNNER_IMAGE` |
| `ANALYZER_MCP_BASE_URL` | `VIBESIM_AGENT_ANALYZER_BASE_URL` |
| `VIBESIM_MANAGED_BACKEND_URL` | `VIBESIM_AGENT_MANAGED_BACKEND_URL` |
| `OPENROUTE_KEY` | `OPENROUTER_API_KEY` |

Set `VIBESIM_AGENT_MAIN_DIR`, `VIBESIM_AGENT_BIND` and `VIBESIM_AGENT_PORT`
explicitly. Replace `uvicorn backend.app:app` with
`uv run --frozen python -m vibesim_agent serve`. Remove retired Agent keys from
the service environment; adding new keys alongside them fails startup.
`python -m vibesim_agent env-reference` supplies the full supported key list.
Preserve naming credentials when replacing `OPENROUTE_KEY`: an existing nonblank
`OPENROUTER_API_KEY` takes precedence; otherwise carry over the old value, then
unset the retired key. The local candidate environment performs this fallback
without writing the credential to disk.

The new optional `serve --startup-config /absolute/deployment/startup.json` entry
can own the reviewed shutdown and automatic migration sequence. Its configuration
and the corresponding read-only `selected-root` command are described in
[Managed Startup Entry](migration-v1.md#managed-startup-entry). When adopting this
entry, start Analyzer only after selection succeeds, with the returned registry
path and the same provider configuration. Keep first-start logs outside the
unpublished target. The local candidate scripts use this shared startup configuration;
they have been reviewed but have not been installed or executed.

The old launch script runs `scripts/with_claude_env.py`. The new package does
not discover shell authentication itself. Preserve that wrapper until explicit
provider configuration replaces its behavior, or supply the complete standard
Claude authentication and endpoint variables before starting the process.
Do not copy credentials into scripts, migration reports or commits. Generate
migration scopes using the same profile paths and endpoints as the final service.

The local candidate scripts are in `tmp/agent-refactor-deployment/` at the
workspace root. They target `wt-agent-refactor`, the existing `VibeSim` main
checkout. Both services require the same absolute `VIBESIM_STARTUP_CONFIG` path;
Agent performs managed startup and Analyzer obtains the published target through
`selected-root`. They retain the Claude wrapper and existing ports, use the root
`.env` for temporary/cache paths, and select the accepted immutable runner image.
Analyzer selects the accepted absolute executable and verifies its SHA256;
conflicting image or binary overrides are rejected. An arbitrary existing binary
may expose a different API prefix. They are not installed or executed.
The existing `agent-workspaces/services/claude.env`, when present, remains an
external deployment asset; candidate logs go to
`tmp/agent-refactor-deployment/logs/`, outside both state roots. Preserve
that credential file without copying its contents into the migration record.
The companion launcher changes must be integrated into the selected source before
building the final image; rebuilding does not update old workspace copies.

A private rehearsal with a separately built Analyzer has verified that both
services select the migrated root, and Agent archive/restore changes update the
Analyzer catalog without restarting it or changing the resource ID. The test used
a synthetic prediction catalog fixture. It does not establish numerical analysis,
provider resume or acceptance of the production ports, bridges and final image.

A separate private migration/resume rehearsal has passed for both GPT roles:
two calls per role kept the original session IDs, recalled exact baseline markers
and appended to the original transcripts across application and container recreation.
All historical sessions, including failed Claude sessions, were preserved by the
conversion. A later GPT turn also passed through the HTTP ASGI application and
real provider/container delegation: implementer created a scoped file, orchestrator
resumed its session to finish, and history/SSE/replay agreed. Only the test file
and two required conversation records changed. These are GPT-only checks;
Claude acceptance was outstanding at that point; the later results above close it.
A separate real TCP/Uvicorn GPT test has
verified incremental SSE and targeted cancellation of an observed live tool process,
including process exit before container removal, session retention, history/replay
and repeated cancellation. A follow-up real-network disconnect/reconnect test also
passed: the same turn/process survived closing the message stream, GET stream
replayed the original event prefix, and cancellation completed without a new
message submission. Those earlier Claude baseline attempts returned
`503 No available accounts` from the configured gateway; the later acceptance
above uses the user's official `claudeme` connection.

## Image Evidence

The observed backend script selects `vibesim-ui-codex-runner:kanzhu-claude`,
image `64d658b3b7f333d09ab53737d1552c13a3e8ac4fd0bdca5fd57e6eb9e3dac731`.
The real build smoke test and new ContainerManager acceptance used
`vibesim-ui-codex-runner:kanzhu`, image
`9c9c0826ff764ac018b008ef303bb057a6aa83d92cbfc996e63be4bc2bbff7a9`.
Their version and lock labels match; their image IDs do not. Acceptance of one
does not validate the other. Record the accepted final image ID before cutover.

Two inspected running conversation containers used other immutable image IDs:

| Container Suffix | Image SHA256 |
| --- | --- |
| `w_15682108fec5-83b4a53c323f` | `23cda76bb520da4d6baef463334ed43ae5c9906eb23e05789dd867a714bb6b5a` |
| `w_0a324903b8df-441877f79e3c` | `277e117ed58d7ef9bc7f57976806fefc1001c97313650e92677cc8a187b91e89` |

Preserve old image IDs and per-container configuration with the deployment backup.
Do not infer a running container's image from the current service script or tag.

## Switch And Rollback

The workspace owner coordinates the switch. Freeze the accepted Agent and
launcher commits, image ID, provider scopes, target paths and service scripts.
Record the operator and time in the local execution log. Stop admission, drain
or cancel turns and jobs, then stop all owned writers before the final snapshot.
The new state root changes container ownership names; new startup recovery will
not adopt or clean up old containers. Confirm old runtime processes have exited
using the old deployment's ownership information before asserting quiescence.
Follow [Offline Workspace Migration](migration-v1.md); never initialize or run
the new schema against the original state. Start only after conversion and
file verification succeed, with Analyzer reading the same migrated registry.

Before reopening writes, verify historical reads, SSE replay, citations, provider
resume and container-to-host managed callbacks. If this fails, stop the new
services and restore the old deployment against its untouched original state.
The simple rollback window ends when user writes resume. After that point,
preserve the new state and reconcile those writes through a verified reverse
conversion or forward repair; never restore an old snapshot over new data.
The candidate checkout contains only the new implementation. The independently
running old checkout and its source archive remain available; retire those deployment
assets only after acceptance and observation. Keep state backups through the rollback window.

## Current Stop Audit

The final read-only inventory found seven workspaces with no running turns. All
historical provider sessions use GPT; their configured models are `gpt-5.6-sol`
and `gpt-5.6-luna`. Existing callback aliases remain necessary for copied launchers.

The shutdown audit currently rejects two non-Agent containers, `areal_banking_prod`
and `areal_keepalive`, with writable mounts of all `/raid`. Their ownership is not
established by the Agent deployment. No service or container was stopped, and no
production migration target was created. Checking current open files would not
prove that these containers cannot write later; their owners must resolve this
mount scope before recapturing the audit. Do not bypass this rejection.

The retained legacy source is commit `e26ad6d`. Its private archive is
`tmp/agent-cutover-final/legacy-agent.tar`, SHA256
`b7036bc7ff3982681c5382c877365bd6906cf2e6f01f516e9ce80193282e9952`.
This is a source archive, not the still-pending stopped-state data backup.
