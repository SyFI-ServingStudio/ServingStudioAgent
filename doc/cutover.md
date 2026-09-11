# Deployment Cutover

Use this procedure when replacing a legacy Agent deployment. Keep deployment
scripts, provider configuration, logs, audit records and backups outside Git and
outside both state roots. Use durable deployment paths, not disposable temporary
directories. See [Migration](migration-v1.md) for converter and startup formats.

## Prepare

Record the Agent, VibeSim launcher and VibeSimUI revisions, immutable runner image,
dependency lock hashes, Analyzer executable and service configuration. Validate
their integration before admitting production writes. Existing copied launchers
still require the [legacy callback aliases](managed-jobs.md).

Identify every service port, process, supervisor and container that can write the
source state or external repositories. Follow the parent workspace's port and
ownership conventions. Do not stop unrelated services or infer ownership from a
name. Record broad writable mounts as well as direct state mounts. An explicitly
accepted external writer is still able to change files; it is not isolated by
the audit exception.

Prepare provider mappings using the exact profile paths, endpoints and connection
IDs intended for the new service. Keep credentials in their configured sources,
not migration reports or scripts. Supply shell-wrapper credentials explicitly or
retain the required wrapper; Agent does not inspect interactive shell functions.

## Stop And Back Up

Close admission, drain or cancel active work, and stop old backends, runtime
containers, jobs and restart supervisors. Confirm that all relevant writers remain
stopped before copying state. A process inventory or two matching file scans alone
does not prove an instantaneous snapshot.

Create and verify an independent backup of the complete state root, including
SQLite sidecars and provider homes. Back up external repositories and logs
separately where required; the state root does not contain those bytes. Retain
the old source revision, images and service configuration for recovery. A source
archive or migration manifest alone is not a data backup.

Run the offline converter with an absent target, or use the audited managed
cutover entry. Never initialize the old state with `init`, and never serve an
incomplete migration target. Keep first-start logs outside the target.

## Switch And Verify

After successful conversion and selection publication, validate historical reads,
message identities, SSE replay, citations, workspace discovery and managed
callbacks. Verify required provider resumes in isolated fixtures using the
original session IDs; conversion success does not establish resumability.
Run [browser integration checks](browser-acceptance.md) against the matching UI
and Analyzer. Confirm Agent and Analyzer use the same registry.

Before adopting the permanent service configuration, resolve the published target
with the same provider configuration used for cutover:

```bash
uv run --frozen python -m vibesim_agent selected-root \
  --startup-config /deployment/control/startup.json
```

Check the command succeeds, then configure its exact returned absolute path as
`VIBESIM_AGENT_WORKSPACES_ROOT`. Configure Analyzer with
`--workspace-registry /deployment/current-state/registry.json`, substituting that
same validated target. Normal Agent starts use:

```bash
uv run --frozen python -m vibesim_agent serve
```

Do not pass `--startup-config` on every production restart. That entry retains
legacy shutdown checks, including boot ID and audited container identities.
Ordinary startup validates the current schema and ownership without requiring
obsolete deployment containers or a pre-reboot receipt. Ensure the old deployment
cannot restart; ordinary `serve` does not enforce legacy shutdown.

Keep executable release code and mutable workspace data separate. Check release
artifacts when deploying, but do not require an editable external `w_main`
checkout to remain clean or at one commit for normal service restarts.

## Configuration Changes

| Legacy Key | Current Key |
| --- | --- |
| `VIBESIM_WORKSPACES_ROOT` | `VIBESIM_AGENT_WORKSPACES_ROOT` |
| `CODEX_DOCKER_IMAGE` | `VIBESIM_RUNNER_IMAGE` |
| `ANALYZER_MCP_BASE_URL` | `VIBESIM_AGENT_ANALYZER_BASE_URL` |
| `VIBESIM_MANAGED_BACKEND_URL` | `VIBESIM_AGENT_MANAGED_BACKEND_URL` |
| `OPENROUTE_KEY` | `OPENROUTER_API_KEY` |

Remove retired keys rather than leaving both spellings set. Set main directory,
bind address, port, callback URL and Analyzer URL explicitly for the deployment.
Use `python -m vibesim_agent env-reference` for the full supported key list.

## Recovery

Before new writes are admitted, a failed cutover can return to the old deployment
against its untouched state after the new services are stopped. No migration
failure automatically restarts the old deployment.

After new writes begin, preserve the target. Restoring an old snapshot would lose
those writes; use forward repair or a separately verified reconciliation instead.
Keep backups and migration evidence according to the deployment retention policy.

If a migrated deployment still uses the managed entry and a reboot invalidates
its receipt, follow [the ordinary-startup transition](migration-v1.md#recovery-after-a-host-reboot).
Do not modify the recorded boot ID or restart the legacy service solely to obtain
a fresh receipt. Partial or unpublished targets require inspection before reuse.
