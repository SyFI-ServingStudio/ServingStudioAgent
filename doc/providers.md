# Named Provider Connections

Copy [providers.yaml](../examples/providers.yaml) to `providers.yaml` in the Agent
checkout and edit your connections. This private file is ignored by Git and
loaded automatically from the Agent checkout, independently of the current
working directory. An explicit `VIBESIM_AGENT_PROVIDERS_FILE` absolute path takes
precedence. The file is required; missing or invalid configuration fails startup.

The example illustrates separate Codex, Claude gateway and Claude personal-login
connections.

The file defines the provider list and all three role defaults. Provider-specific
`VIBESIM_PROVIDER_*` variables are rejected, so a shell's old settings cannot
silently change a named connection. Agent/container environment
settings and the naming credential remain separate. Configuration is read at
startup; changes require a restart. Init, serve and managed migration validation
use the same loader.

Each provider ID names a connection, not a model or container. Use lowercase
letters, digits and underscores. The adapter selects the installed CLI (`codex`
or `claude`); the label is shown in the model picker. Two connections can offer
the same model: the UI sends the connection ID with the model when selecting it.
Legacy model-only requests retain their previous selection behavior.

The current schema uses `default_model`, `default_effort`, and a `models` mapping.
Every model declares its own selectable efforts:

```yaml
default_model: gpt-6-astra
default_effort: high
models:
  gpt-6-astra:
    efforts: [low, medium, high, xhigh, max, ultra]
```

The defaults must name one declared model and an effort supported by every model
in that provider. The old `model`, `effort`, and `models: [name, ...]` form is
rejected. A GLM gateway does not inherit another model's choices; declaring choices
does not prove the endpoint supports them.
A Codex profile's cached model catalog may enrich labels and service tiers but
cannot add undeclared models or change configured efforts. `defaults` must select
a defined connection for `orchestrator`, `implementer` and `assistant`.

## Credentials

`environment` maps a CLI credential variable to the name of a host environment
variable. For example, `ANTHROPIC_AUTH_TOKEN: WORK_CLAUDE_TOKEN` reads the token
from `WORK_CLAUDE_TOKEN`. Alternatively, an explicit value can be stored in this
ignored private YAML: `ANTHROPIC_AUTH_TOKEN: {value: "your-token"}`. Literal values
are loaded as masked secrets and never returned by the model catalog. Keep this
file private (mode 600); never put real tokens in tracked examples. Shell functions
are not invoked or parsed by the service. Credentials assigned only inside an
interactive wrapper must be supplied separately in the service environment.
Missing values make that connection unavailable and never fall back to another
connection's credential. Source references and CLI credential names are removed
from other adapters' process environments and from general Docker operations.

For Codex, `home` supplies selected configuration/authentication files, not host
conversation history. Its `config.toml` defines the endpoint. An environment-based
credential must be explicitly mapped to the variable named by that config's
`env_key`. Claude accepts either one authentication environment reference, or a
`home` containing `.credentials.json`. These two sources are mutually exclusive.
Claude's `base_url` is explicit; omitting it selects the CLI's default endpoint,
without inheriting the host's gateway setting.

Claude home authentication copies only `.credentials.json` into the isolated
runtime home. CLI-refreshed credentials remain there; a changed host credential
file supplies a new copy on the next prepare. Personal history/settings are not
copied and refreshed credentials are not written back into the host login.
Validate login, original-session resume and token refresh separately: a successful
turn or a resumed session does not demonstrate an actual OAuth refresh. Use an
isolated conversation and retain the profile path and endpoint across restart
checks. See [browser and execution validation](browser-acceptance.md).

## Amazon Bedrock

A Claude connection with a `bedrock` block runs Claude Code against Amazon
Bedrock (`CLAUDE_CODE_USE_BEDROCK=1`) in the declared `region`. Model IDs are
Bedrock IDs or inference profiles, such as `us.anthropic.claude-opus-5-5`, and
must be enabled for the account in that region. `base_url` and a Claude login
`home` are rejected for Bedrock connections.

Credentials come from exactly one source:

- `profile`: a profile with static keys in `credentials_file` (default
  `~/.aws/credentials`). Each runtime preparation copies only that profile's
  access key, secret and optional session token into the isolated runtime home,
  so a rotated host key reaches the next turn and other profiles never enter a
  runner. SSO and `credential_process` profiles have no static keys there and
  are rejected; export temporary keys or use an API key instead.
- `environment`: `AWS_BEARER_TOKEN_BEDROCK` alone, or `AWS_ACCESS_KEY_ID` with
  `AWS_SECRET_ACCESS_KEY` and optionally `AWS_SESSION_TOKEN`, using the same
  reference or inline-value forms as other credentials.

Optional `small_fast_model` sets Claude Code's background model
(`ANTHROPIC_DEFAULT_HAIKU_MODEL`) when the CLI default is not enabled in the
account. Region and profile are part of the session identity; key rotation is
not.

The operator's own Claude Code and AWS shell settings (`CLAUDE_CODE_USE_BEDROCK`,
`AWS_*`, `ANTHROPIC_MODEL` and `ANTHROPIC_DEFAULT_*_MODEL`) are removed from every
provider process and Docker operation, so a host configured for Bedrock does
not silently switch another Claude connection's backend.

## Sessions And Compatibility

Runtime homes remain separate by conversation, role and session scope. Models,
labels, role defaults and token values do not define session identity. Connection
ID, adapter, endpoint and applicable profile home do. Optional `session_identity`
can distinguish accounts behind the same endpoint without storing account secrets.
Keep it stable across token rotation; change it when changing the account.

To adopt YAML for existing history, keep IDs `gpt`, `deepseek` and `claude` and
their existing adapter/backend/profile paths. Omit `session_identity` unless
intentionally changing compatibility. Equivalent declarations preserve the old
scope hash. Renaming a connection or changing account identity is not a transparent
rename: historical sessions stay bound to their original identity, and populated
conversations reject incompatible runtime changes. Keep the old connection while
its sessions are needed. No database schema migration is introduced by YAML.

Duplicate keys, unknown fields, invalid role references and YAML anchors/aliases
are rejected. Diagnostics do not include credential values or the input document.
