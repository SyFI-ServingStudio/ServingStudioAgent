# Named Provider Connections

Set `VIBESIM_AGENT_PROVIDERS_FILE` to an absolute YAML path before starting the
Agent. [providers.yaml](../examples/providers.yaml) illustrates separate Codex,
Claude gateway and Claude personal-login connections. Without that variable,
the existing environment-only `gpt`, `deepseek` and `claude` setup is unchanged.

The file replaces the provider list and all three role defaults. Provider-specific
`VIBESIM_PROVIDER_*` overrides are rejected when YAML is selected, so a shell's old
defaults cannot silently change a named connection. Agent/container environment
settings and the naming credential remain separate. Configuration is read at
startup; changes require a restart. Init, serve and managed migration validation
use the same loader.

Each provider ID names a connection, not a model or container. Use lowercase
letters, digits and underscores. The adapter selects the installed CLI (`codex`
or `claude`); the label is shown in the model picker. Two connections can offer
the same model: the UI sends the connection ID with the model when selecting it.
Legacy model-only requests retain their previous selection behavior.

`model`, `effort` and optional `service_tier` choose defaults. Model capability
validation still applies. Named connections expose only `model` unless an explicit
`models` list is supplied. That list must be nonempty, contain unique model IDs,
and include the default `model`. For example, `models: [claude-sonnet-5, claude-opus-5]`
enables both on a connection known to serve them; a GLM gateway does not inherit
Anthropic's model list. For an unknown model, the configured effort is the fallback
capability; declaring it does not prove the remote endpoint supports it.
A Codex profile's cached model catalog enriches the declared models' capabilities
but cannot add undeclared models. `defaults` must select a defined connection for
`orchestrator`, `implementer` and `assistant`.

## Credentials

`environment` maps a CLI credential variable to the name of a host environment
variable. For example, `ANTHROPIC_AUTH_TOKEN: WORK_CLAUDE_TOKEN` reads the token
from `WORK_CLAUDE_TOKEN`; the YAML never contains the token itself. Shell functions
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
copied and refreshed credentials are not written back into the host login. This
file-based path has local fixture coverage; real provider acceptance is separate.

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
