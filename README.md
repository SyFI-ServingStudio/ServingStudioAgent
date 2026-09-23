<p align="center">
  <img src="doc/assets/servingstudio-symbol.svg" alt="ServingStudio logo" width="64">
</p>

<h1 align="center">ServingStudio Agent</h1>

<p align="center">
  <strong>An agent runtime for running and analyzing LLM serving experiments.</strong>
</p>

<p align="center">
  <a href="#key-features">Features</a> ·
  <a href="#execution-modes">Execution modes</a> ·
  <a href="#repository-map">Repository map</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="doc/architecture.md">Architecture</a>
</p>

---

ServingStudio Agent connects Codex and Claude Code to persistent experiment workspaces.
Describe a serving question, let the agent prepare and run an experiment, and
follow its analysis back to the results. Conversations, provider sessions, and
experiment files persist across turns.

The Agent manages **execution and conversation history**.
[ServingStudio Sim](https://github.com/SyFI-ServingStudio/ServingStudioSim) supplies the simulator and
Analyzer; [ServingStudio UI](https://github.com/SyFI-ServingStudio/ServingStudioUI) provides the
browser interface.

## 📣 News

- **September 2026:** ServingStudio Agent is now available!

---

<a id="key-features"></a>

## ✨ Key features

- 🔌 **Multiple provider connections.** Use named Codex and Claude connections
  for different accounts or endpoints. Each connection declares its models,
  per-model reasoning efforts, and defaults in a private YAML file.
- 📁 **Persistent workspaces.** Keep source code, experiments, and logs together.
  Conversations in a workspace share its files and each has its own isolated
  provider sessions. A sandboxed copy also gets its own Docker container; a git
  worktree runs on the host instead, where kernel profiling and Slurm work.
  See [Execution modes](#execution-modes) for what that trade costs.
- 🧩 **Flexible execution.** Work with a single assistant or use an orchestrator
  and implementer with separate provider selections and resumable sessions.
- 💬 **Durable conversations.** Stream progress, revisit history, reconnect
  without losing recorded events, and cancel active work. Startup recovery
  reconciles interrupted turns.
- 🔎 **Traceable experiments.** Track simulation, timing prediction, kernel
  profiling, and kernel measurement jobs. Analyzer resource references connect
  an agent's findings to the numerical results behind them.

---

<a id="execution-modes"></a>

## 🏠 Execution modes

A workspace decides where its turns run. There is no per-turn switch.

| Workspace kind | Repository | Turns run | Good for |
| --- | --- | --- | --- |
| **Sandboxed copy** | a copy of the tracked files | in a per-conversation Docker container | experimenting safely; sharing a demo |
| **Git worktree** / main checkout | a real branch of your checkout | on this machine, with the tree as the working directory | kernel profiling, Slurm jobs, anything needing the host's tools |

The container's isolation is real, but so is its cost: most of this project's
kernel profiling needs `docker`, a writable submodule, or an environment under
your home directory, and none of those exist inside the runner. There is no
Slurm client in the image either. A worktree workspace is how an agent reaches
them.

> [!WARNING]
> **Host execution mode is trusted.** A host turn runs as you, in a real branch
> of your checkout, and the two CLIs are not equally constrained.
>
> **Codex** runs under a named permission profile that does hold: writes to
> `$HOME` and to other worktrees' working trees are refused. It has two openings
> that were chosen on purpose — access to `/var/run/docker.sock`, which is
> equivalent to root, and escalations approved by a model reviewer, which run
> with no sandbox at all. It stops mistakes, not a determined escape.
>
> **Claude** has no OS boundary in this mode. Its permission mode is a model
> classifier, and `Bash` goes around the tool allowlist.
>
> Committing from a worktree also means write access to the git common
> directory, which every worktree shares — shared refs, objects, and other
> branches included. Do not point host mode at a checkout you would not hand to
> the model outright.

Set `VIBESIM_AGENT_WORKTREE_ROOT` to enable worktree workspaces; leaving it
unset disables the feature, and the browser says so rather than hiding it.

---

<a id="repository-map"></a>

## 🗂️ Repository map

```text
ServingStudioAgent/
├── vibesim_agent/
│   ├── api/          HTTP routes and streamed events
│   ├── services/     Conversations, turns, jobs, and workspace lifecycle
│   ├── domain/       Roles, events, and shared data contracts
│   ├── providers/    Codex and Claude CLI adapters
│   ├── runtime/      Container and host execution, mounts, and provider homes
│   ├── storage/      SQLite persistence and workspace registry
│   └── prompts/      Role instructions and response contracts
├── docker/           Runner image definition
├── scripts/          Image builds and acceptance checks
├── examples/         Public provider configuration examples
├── tools/            State migration and deployment auditing
├── tests/            Service, storage, and provider tests
└── doc/              Architecture, configuration, and operations
```

---

<a id="quick-start"></a>

## 🚀 Quick start

> [!TIP]
> **Recommended: set up through [ServingStudio](https://github.com/SyFI-ServingStudio/ServingStudio).**
> It pins compatible Agent, simulator, and UI revisions and provides shared
> build and service commands. Follow its
> [setup guide](https://github.com/SyFI-ServingStudio/ServingStudio/blob/main/reproduce.md)
> for the complete application.

### Requirements

- **Linux**, **Git**, and **Python 3.12** managed by **uv**.
- **Docker Engine** with daemon access and a prepared Agent runner image.
- A **ServingStudio Sim checkout** and a writable directory for persistent workspace state.
- At least one configured **Codex or Claude connection** with valid credentials.
- An **Analyzer service** reachable from the runner containers.
- NVIDIA Container Toolkit and compatible GPUs when experiments require GPU
  profiling. Cached simulations do not require GPU execution.

### 1. Install and configure

For a standalone Agent checkout:

```bash
git clone https://github.com/SyFI-ServingStudio/ServingStudioAgent.git
cd ServingStudioAgent
uv sync --frozen

cp examples/providers.yaml providers.yaml
chmod 600 providers.yaml
```

Edit `providers.yaml` to keep the connections you use, configure their credentials,
and select the default connection for each role. The file is ignored by Git and
loaded automatically from this checkout. Missing or invalid configuration fails
startup. See [provider configuration](doc/providers.md) for the schema.

Inspect the supported service and runner settings:

```bash
uv run --frozen python -m vibesim_agent env-reference
```

Set `VIBESIM_AGENT_MAIN_DIR` to the absolute ServingStudio Sim checkout path and
`VIBESIM_AGENT_WORKSPACES_ROOT` to an absolute state path outside that checkout.
For fresh initialization, the state directory must not already exist.

Configure the bind address and a free port, plus
`VIBESIM_AGENT_MANAGED_BACKEND_URL` and `VIBESIM_AGENT_ANALYZER_BASE_URL`.
Both URLs must be reachable from the runner containers. The
[service operations guide](doc/service-operations.md) covers configuration
and runner image requirements.

### 2. Prepare the runner and start

With deployment settings and provider configuration in place:

```bash
./scripts/build-runner-image.sh
uv run --frozen python -m vibesim_agent init
uv run --frozen python -m vibesim_agent serve
```

Build the image once, initialize fresh state once, and use `serve` for subsequent
starts. Serving does not build the image or launch the browser UI. Configure
Analyzer with the state directory's `registry.json` and connect ServingStudio UI to this
service. The workspace setup guide provides coordinated startup commands.

Existing deployments should follow the [migration guide](doc/migration-v1.md)
and [cutover guide](doc/cutover.md) before changing their state directory.

### 3. Start an experiment

In ServingStudio UI, choose a workspace, create a conversation, and select your provider
and execution mode. A first task could be:

> Run the included Llama 3 8B smoke simulation and summarize request completion,
> throughput, and latency.

The conversation records progress and links managed jobs to Analyzer results.
Workspace files and logs remain available after the conversation container stops.

---

## 📚 Documentation

| Guide | Topics |
| --- | --- |
| [Provider configuration](doc/providers.md) | Accounts, credentials, models, and efforts |
| [Architecture](doc/architecture.md) | Execution, ownership, and persistence |
| [Service operations](doc/service-operations.md) | Images, settings, startup, and runtime limitations |
| [Agent API](SKILL.md) | Conversations, tools, streaming, and cancellation |
| [Managed jobs](doc/managed-jobs.md) | Launcher callbacks and result linkage |
| [Migration](doc/migration-v1.md) | Converting existing state |
| [Browser acceptance](doc/browser-acceptance.md) | End-to-end integration checks |

Run the local service tests with:

```bash
uv run --frozen python -m unittest discover -s tests -v
```

These cover service behavior, storage, and provider parsing. Real provider
execution and GPU profiling require the corresponding integration environment.
