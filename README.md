<p align="center">
  <img src="doc/assets/vibesim-logo.svg" alt="VibeSim logo" width="64">
</p>

<h1 align="center">VibeSim Agent</h1>

<p align="center">
  <strong>An agent runtime for running and analyzing VibeSim experiments.</strong>
</p>

<p align="center">
  <a href="#key-features">Features</a> ·
  <a href="#repository-map">Repository map</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="doc/architecture.md">Architecture</a>
</p>

---

VibeSim Agent connects Codex and Claude Code to a persistent VibeSim workspace.
Describe a serving question, let the agent prepare and run an experiment, and
follow its analysis back to the results. Conversations, provider sessions, and
experiment files persist across turns.

The Agent manages **execution and conversation history**.
[VibeSim](https://github.com/SyFI-VibeSim/VibeSim) supplies the simulator and
Analyzer; [VibeSimUI](https://github.com/SyFI-VibeSim/VibeSimUI) provides the
browser interface.

## 📣 News

- **September 2026:** VibeSim Agent is now available!

---

<a id="key-features"></a>

## ✨ Key features

- 🔌 **Multiple provider connections.** Use named Codex and Claude connections
  for different accounts or endpoints. Each connection declares its models,
  per-model reasoning efforts, and defaults in a private YAML file.
- 📁 **Persistent workspaces.** Keep source code, experiments, and logs together.
  Conversations in a workspace share its files, while each conversation has
  its own Docker container and isolated provider sessions.
- 🧩 **Flexible execution.** Work with a single assistant or use an orchestrator
  and implementer with separate provider selections and resumable sessions.
- 💬 **Durable conversations.** Stream progress, revisit history, reconnect
  without losing recorded events, and cancel active work. Startup recovery
  reconciles interrupted turns.
- 🔎 **Traceable experiments.** Track simulation, timing prediction, kernel
  profiling, and kernel measurement jobs. Analyzer resource references connect
  an agent's findings to the numerical results behind them.

---

<a id="repository-map"></a>

## 🗂️ Repository map

```text
VibeSimAgent/
├── vibesim_agent/
│   ├── api/          HTTP routes and streamed events
│   ├── services/     Conversations, turns, jobs, and workspace lifecycle
│   ├── domain/       Roles, events, and shared data contracts
│   ├── providers/    Codex and Claude CLI adapters
│   ├── runtime/      Docker execution, mounts, and provider homes
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
> **Recommended: set up through [VibeSimWorkspace](https://github.com/SyFI-VibeSim/VibeSimWorkspace).**
> It pins compatible Agent, simulator, and UI revisions and provides shared
> build and service commands. Follow its
> [setup guide](https://github.com/SyFI-VibeSim/VibeSimWorkspace/blob/main/reproduce.md)
> for the complete application.

### Requirements

- **Linux**, **Git**, and **Python 3.12** managed by **uv**.
- **Docker Engine** with daemon access and a prepared Agent runner image.
- A **VibeSim checkout** and a writable directory for persistent workspace state.
- At least one configured **Codex or Claude connection** with valid credentials.
- An **Analyzer service** reachable from the runner containers.
- NVIDIA Container Toolkit and compatible GPUs when experiments require GPU
  profiling. Cached simulations do not require GPU execution.

### 1. Install and configure

For a standalone Agent checkout:

```bash
git clone https://github.com/SyFI-VibeSim/VibeSimAgent.git
cd VibeSimAgent
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

Set `VIBESIM_AGENT_MAIN_DIR` to the absolute VibeSim checkout path and
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
Analyzer with the state directory's `registry.json` and connect VibeSimUI to this
service. The workspace setup guide provides coordinated startup commands.

Existing deployments should follow the [migration guide](doc/migration-v1.md)
and [cutover guide](doc/cutover.md) before changing their state directory.

### 3. Start an experiment

In VibeSimUI, choose a workspace, create a conversation, and select your provider
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
