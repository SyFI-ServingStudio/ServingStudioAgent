# VibeSim Agent

VibeSim Agent runs Codex and Claude Code in Docker and owns durable workspaces,
conversations, turns, provider sessions and managed jobs. VibeSimUI is the browser
application; Rust Analyzer owns numerical results and their catalogs.

`run.sh` starts the `vibesim_agent` service. The browser application lives in
VibeSimUI. See [Architecture](doc/architecture.md) for component ownership and
[Deployment Cutover](doc/cutover.md) when replacing a legacy deployment.

For workspace setup, use the parent workspace's `README.md` and `reproduce.md`.
Run commands below from this checkout with Python 3.12 and `uv`, after sourcing
the workspace-root `.env`.

## Start The Service

The `vibesim_agent` package has explicit initialization, migration and serving
commands. Starting the service does not build the UI or a runner image.

```bash
uv run --frozen python -m vibesim_agent env-reference
```

For fresh state, set `VIBESIM_AGENT_MAIN_DIR` to the absolute VibeSim checkout
path and `VIBESIM_AGENT_WORKSPACES_ROOT` to a new absolute directory outside that
checkout. Initialization creates the external `w_main` descriptor, current
SQLite schema and registry index; it does not copy or modify the main checkout
or its logs. Any existing state directory is rejected, including an empty one.

```bash
uv run --frozen python -m vibesim_agent init
```

Before serving, select and check the bind address and port using the workspace
`reproduce.md` per-user convention. Set `VIBESIM_AGENT_BIND`,
`VIBESIM_AGENT_PORT`, `VIBESIM_AGENT_MANAGED_BACKEND_URL` and
`VIBESIM_AGENT_ANALYZER_BASE_URL` for that deployment; callback and Analyzer URLs
must be reachable from the runner. Runner images must already be built.

```bash
uv run --frozen python -m vibesim_agent serve
```

`./run.sh` is a shorthand for this command and forwards its arguments, including
`--startup-config` for a reviewed migration. It requires the same environment.

The service uses one process. Its factory, `vibesim_agent.bootstrap:create_application`,
checks all workspace databases and acquires the state directory lock before
writing generated prompts. Lifespan startup recovers interrupted turns before
accepting requests; shutdown drains activity before releasing ownership.
Existing databases with an older format require managed startup or offline
migration, never `init`.
Retired Agent environment keys are rejected by name, without printing their values.
Provider connections, models, per-model efforts, and defaults are declared in the
required `providers.yaml`; per-conversation selection remains available through the API.

For multiple accounts or endpoints using the same CLI, copy
`examples/providers.yaml` to `providers.yaml` in this Agent checkout and edit it.
The private file is ignored by Git and loaded automatically, regardless of the
working directory. `VIBESIM_AGENT_PROVIDERS_FILE` can select a different absolute
path explicitly. The file is required; a missing or invalid file fails startup.
Named connections select
their own profile or credential reference and all three role defaults; see
[Named Provider Connections](doc/providers.md) and [example YAML](examples/providers.yaml).

For a legacy deployment, `serve --startup-config /absolute/startup.json` is the
cutover entry: it validates the audited shutdown scope, migrates into an independent
target and publishes a selection record. Once migration is verified, resolve that
target with `selected-root --startup-config /absolute/startup.json`, set
`VIBESIM_AGENT_WORKSPACES_ROOT` to the returned absolute path, and use ordinary
`serve` for subsequent starts. Configure Analyzer with the same target's
`registry.json`. Both cutover commands require the same provider configuration.
Do not permanently couple normal service restarts to the legacy shutdown receipt.
See [managed startup configuration](doc/migration-v1.md#managed-startup-entry).

## Runner Image Preparation

The refactor has an explicit image build entry point. Use a separate image tag
for rehearsal and pass that same tag to the rehearsal backend:

```bash
VIBESIM_RUNNER_IMAGE="vibesim-agent-runner:${USER}-refactor" \
  ./scripts/build-runner-image.sh
```

It uses the same `VIBESIM_AGENT_MAIN_DIR` and `VIBESIM_RUNNER_*` runtime
configuration as the new backend, including image tag, UID/GID and user. The
source must be a Git repository root with `uv.lock`. It copies tracked working
files and required initialized submodules into a private build context; newly
added source files must be tracked before building. It does not include
untracked workspace state or supply provider credentials as build arguments.

Additional build-only overrides are `VIBESIM_RUNNER_CUDA_IMAGE`,
`VIBESIM_RUNNER_UV_IMAGE`, `VIBESIM_RUNNER_CODEX_NPM_PACKAGE`,
`VIBESIM_RUNNER_CLAUDE_NPM_PACKAGE`, `VIBESIM_RUNNER_NODE_VERSION`,
`VIBESIM_RUNNER_NODE_ARCH`, and `VIBESIM_RUNNER_RUST_TOOLCHAIN`.
`VIBESIM_RUNNER_SKIP_IMAGE_TEST=1` explicitly skips post-build acceptance;
the default is `0`, which runs the build smoke test against the same source
directory. A build or smoke failure exits unsuccessfully.

The current Dockerfile provides `/home/<runner-user>`, `/opt/vibesim-venv`,
`/opt/vibesim-uv-cache`, and `DG_USE_LOCAL_VERSION=0`. The new entry rejects
incompatible runtime overrides before building; other images may support them.
The new entry uses `docker/runner.Dockerfile` and `scripts/test-runner-image.sh`,
with the default image `vibesim-agent-runner:<runner-user>` and runner version
`prebuilt-agent-runner-v12`. Image labels use `org.vibesim.agent.*`.
Node.js is pinned to `v22.23.2` to satisfy the pinned Claude CLI's Node >=22
requirement; npm rejects an incompatible engine during Claude installation.
The previous Dockerfile and build commands are retained in the old deployment
snapshot, not in this checkout.
The smoke script reads the same new configuration; `timing` mode requires a
nonempty `VIBESIM_RUNNER_GPUS`. It reads the GPU model inside the container and
runs a private copy of the timing preset with that model and fresh output paths.
Mixed visible GPU models are rejected; select same-model devices with
`VIBESIM_RUNNER_GPUS`. Profiling retains the launcher's idle-device checks.
Failed smoke runs that produced logs retain their private workspace and print
its path for diagnosis; successful runs remove the private copy.
No image is built at Agent
service startup. The image retains dependency caches; first-party simulator
and Analyzer binaries still build in the workspace when the launcher needs them.
The image does not currently provide Docker for nested profiling containers.
A cold profile cache that needs a container-backed kernel, such as the preset's
`kv_cache_append:vllm_cuda`, fails with `docker is required for container profiling`.
GPU visibility and prebuilt dependency caches alone do not satisfy that runtime
requirement. Validate the intended cold-cache profiling path separately from
image startup and warm-cache simulation.

## State And Migration

A workspace owns one repository and logs root shared by its conversations.
Managed workspaces copy tracked source files from `VIBESIM_AGENT_MAIN_DIR`;
`w_main` references that checkout directly. Each workspace has a descriptor and
SQLite database. `registry.json` lets Analyzer discover active workspaces.
Conversation containers and role/provider homes are isolated; deleting a
container leaves durable workspace data intact.

The new storage format makes role and provider session ownership explicit.
Migration converts that representation while preserving historical content.
It copies into an independent target, retains original files and the legacy
provider-home tree, archives the original descriptors and databases, and verifies
typed database values, references and file contents. External repositories and
logs keep their original locations in production mode.

Automatic migration runs at managed service startup, before requests are accepted,
when `serve --startup-config ...` detects supported legacy state. It stops the
explicitly audited old deployment, verifies conversion, and publishes the target
selection. After cutover, configure ordinary `serve` with the validated target.
Plain `serve` rejects an old schema;
`init` is only for an absent state directory.

This preserves data; it does not by itself prove that a provider can resume every
historical CLI session. Verify each required provider and role against an isolated
historical session. Mixed or unknown formats, changed deployment
identities and incomplete targets fail without overwriting them. A crash can
leave an incomplete unpublished target requiring inspection. Once users write to
the new state, reverting to an old snapshot would lose those new writes.

See [migration and managed startup](doc/migration-v1.md) for configuration,
verification and recovery, and [deployment cutover](doc/cutover.md) for the
coordinated Agent/Analyzer switch and rollback window.

## Conversations And Integrations

The API retains workspace-scoped routes under `/api/agent/v1/workspaces` for
conversation creation, history, messages, SSE replay and cancellation.
Browser files use `/api/agent/v1/file`, `/api/agent/v1/file/meta` and
`/api/agent/v1/file/list`, with `workspace_id` in the query. Tools artifacts use
`/api/agent/v1/tools/workspaces/{workspace_id}/artifacts`.
The compatibility conversation index is read-only. See [Agent API](SKILL.md)
for requests, tools authentication and runtime selection.

`orchestrated` mode uses separate orchestrator and implementer roles; `single`
mode uses one assistant. Each role selects a registered provider and retains its
own resumable session. Mode and runtime changes remain subject to conversation
compatibility checks. Browser compatibility names such as `codex_runtime` and
`codex-backends` remain supported.

Managed simulation, timing prediction, kernel profile and kernel measurement jobs
share the new registration/status endpoints. Existing workspace launchers retain
their legacy callback aliases. These callbacks use a turn-scoped capability;
they do not use the tools API token. Deploy the companion VibeSim launcher changes
with this service. See [Managed Job Callbacks](doc/managed-jobs.md).

## Configuration

`env-reference` generates the supported environment reference from the same
settings definitions used at startup. Configuration is loaded explicitly;
retired Agent keys fail startup instead of silently falling back.

| Configuration | Namespace |
| --- | --- |
| Source, state, bind, callbacks, tools token | `VIBESIM_AGENT_*` |
| Image, container identity, GPU selection | `VIBESIM_RUNNER_*` |
| Provider connections, models, efforts, defaults | `providers.yaml` |

Use `VIBESIM_AGENT_API_TOKEN` for the tools API. It is not general authentication
for every browser route. Provider credentials are runtime inputs, never image
build arguments. The new service does not discover Claude shell credentials;
supply the configured authentication and endpoint environment, or retain the
deployment's credential wrapper as described in the cutover guide.
Optional automatic naming uses `OPENROUTER_API_KEY`.

## Code And Validation

| Directory | Responsibility |
| --- | --- |
| `vibesim_agent/api/` | HTTP validation, routing and SSE transport |
| `vibesim_agent/services/` | Turns, conversations, jobs and workspace lifecycle |
| `vibesim_agent/domain/` | Shared roles, events, requests and results |
| `vibesim_agent/providers/` | Provider registry, CLI adapters and event parsing |
| `vibesim_agent/runtime/` | Containers, mounts, role homes and process control |
| `vibesim_agent/storage/` | SQLite persistence and workspace registry |
| `vibesim_agent/prompts/` | Role contracts, rendering and compatibility fingerprints |
| `tools/` | Offline migration, startup selection and deployment auditing |

Run the local suite with:

```bash
uv run --frozen python -m unittest discover -s tests -v
```

Local tests cover storage conversion, startup recovery, provider parsing and HTTP
behavior. They do not establish real provider availability, GPU profiling or
production cutover. Frozen legacy fixtures allow the new suite to run without
importing a legacy backend. Legacy snapshot tools require a separately retained
source checkout; they do not inventory the current router-based application.

[Browser acceptance](doc/browser-acceptance.md) describes integration checks for
VibeSimUI, Agent and Analyzer.
