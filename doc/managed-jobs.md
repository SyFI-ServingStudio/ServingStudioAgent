# Managed Job Callbacks

Agent accepts all four job kinds at
`POST /api/agent/v1/internal/jobs/register`. Requests use the current turn's
Bearer capability, not the tools API token. Register before creating official
artifacts; a failed registration must not become an unmanaged invocation.

| Field | Meaning |
| --- | --- |
| `job_kind` | `simulation`, `timing_predict`, `kernel_profile`, or `kernel_measure` |
| `artifact_root` | A directory below this workspace's logs root |
| `analyzer_resource_id` | Required for the three non-simulation kinds; absent for simulation |
| `run_count` | Simulation run count, at least 1; defaults to 1 |
| `axes` | Simulation sweep axes; defaults to an empty list |

The equivalent camelCase names (`jobKind`, `artifactRoot`,
`analyzerResourceId`, `runCount`) are also accepted. Non-simulation requests
reject non-default `run_count` or nonempty `axes`. Simulation assigns its own
experiment identity and rejects an explicit Analyzer resource ID.

Responses retain the existing camelCase fields: `jobId`, `approvedRoot`, and
ownership identifiers. Simulation returns `experimentId`; other kinds return
`resourceId` and `analyzerResourceId`. Callers must check the approved path
before writing. Simulation registration retains the existing experiment
metadata and database reconciliation behavior.

Report status to `POST /api/agent/v1/internal/jobs/{job_id}/status` with
`{"status": "running"}` (or `analysis_running`, `ready`, `failed`,
`interrupted`). The stored job kind selects the existing event vocabulary:
simulation continues to emit simulation/analysis/experiment events; other
kinds emit job events. The capability must belong to the job's conversation
and turn, and that turn must still be running. Storage rechecks ownership and
turn state in the update transaction.

## Compatibility During Migration

Both `/api/internal/` and `/api/agent/v1/internal/` continue to provide the
existing `managed-runs/register`, `managed-runs/{job_id}/status`,
`managed-jobs/register`, and `managed-jobs/{job_id}/status` routes. They retain
their original request shapes and simulation versus non-simulation behavior.
New and old routes operate on the same durable jobs. Historical workspace
copies can therefore continue using their existing clients.

The server still ignores client `descriptor` and status `summary` fields,
matching the old backend. They are not a stored result payload. Analyzer
remains the numerical result authority.

The Agent's schema-version-1 context now includes
`"managed_jobs_api": "agent-v1"`. The companion ServingStudio Sim launcher uses this marker
to select the unified endpoint before making requests. Without the marker it
uses the legacy callback family, supporting old backend contexts. An unknown
marker, including explicit null, fails before HTTP. Existing clients ignore the
new field and continue using the retained legacy routes.

The client never retries a failed registration against another endpoint after
a timeout or server error, since the first request may already have created a
job. Image rebuilding alone does not update launcher files in existing
workspaces. The companion changes must be deployed with the Agent; historical
copies continue to use legacy callbacks until separately updated.

The current ServingStudio Sim caller inventory is:

| Caller | Client and behavior to preserve |
| --- | --- |
| `launcher/sweep.py` | `ManagedRun`; single/sweep registration and baseline Analyzer subjects |
| `launcher/timing_predict.py` | `ManagedJob`; timing prediction registration and lifecycle |
| `profiling/cli.py` | `ManagedJob`; kernel profile and measurement registration and lifecycle |

Both updated clients prefer `VIBESIM_MANAGED_JOB_CONTEXT` and fall back to
`VIBESIM_MANAGED_RUN_CONTEXT` only when the preferred variable is empty. Old
`ManagedRun` copies read only the run variable. The Agent currently injects
both. Without either context, developer commands remain
local; simulation writes its stable development experiment metadata. With a
configured context, unreadable or invalid context is fatal. Successful status
callbacks are deduplicated by the clients, while failed requests remain
retryable; typed callbacks with a summary continue to be sent even when the
status repeats.

## Cross-Repository Check

From the Agent checkout, explicitly select the new and baseline ServingStudio Sim source
checkouts when running the callback compatibility check:

```bash
uv run python -m tests.peer_launcher_contract \
  --launcher-root /path/to/updated/ServingStudioSim \
  --legacy-launcher-root /path/to/baseline/ServingStudioSim -v
```

This runs the actual clients through the Agent ASGI app and SQLite, bridging
only urllib's socket transport. It covers four job kinds for new clients with
new context, new clients with legacy context, and legacy clients with new
context. All three use the new Agent server, including its retained legacy
routes; this is not a test of a running old backend, Docker networking, or
provider execution. Ordinary Agent test discovery does not assume external
source checkouts are available.

Inventory launcher copies in active and archived workspaces before retiring a
callback alias. Updating the main checkout or runner image does not update those
copies, and callback compatibility does not validate their container configuration.
