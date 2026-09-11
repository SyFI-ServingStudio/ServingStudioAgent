# Offline Workspace Migration

`tools/migrate_v1_workspaces.py` assembles the old workspace state into a new,
independent directory. It preserves original files, converts databases, maps
historical provider homes, rewrites descriptor locations and rebuilds the index.
It does not stop writers, start a service or establish provider resume compatibility.

```bash
uv run python -m tools.migrate_v1_workspaces OLD_ROOT NEW_ROOT --mapping workspace-mapping.json
uv run python -m tools.migrate_v1_workspaces OLD_ROOT NEW_ROOT --mapping workspace-mapping.json --apply --source-quiesced
```

Without `--apply`, the command inventories files and plans the conversion without
creating the destination. Small database copies in temporary directories keep
SQLite readers from modifying source SHM files. Applying requires
`--source-quiesced`: the caller must already have stopped every old backend,
container, job and other writer. The flag records that assertion; it does not
perform or prove the stop. Plan and apply must use an absent target with an
existing parent outside the source and all external workspace repo/log paths.

The mapping document contains `models` and `families` as described below, plus
`runners`, mapping each exact historical family to `codex` or `claude`. Every
family needs a runner. Scope and runner must match the intended target provider
configuration; the converter cannot infer historical CLI compatibility.

Default `--mode production` preserves original external repo/log locations and
only prepares new state. For `--mode rehearsal`, supply `external_paths` in the
mapping: each external workspace ID maps to `repo_path` and `logs_path`, both
existing absolute paths to independently prepared copies. Every external
workspace needs an override. Originals and source state must not overlap these
copies. Logs must be the real `repo/logs` directory, not a symlink, because the
current container mounts the repo. The converter does not prepare or validate
these external copies; their file fidelity and execution isolation remain
separate acceptance checks.

The new state retains the complete original `codex/` tree and unknown files.
`.migration-v1/original/` stores the old `registry.json` and each workspace's
descriptor, SQLite database and present WAL/SHM/journal files, using their original
relative paths. Conversion operates on separate scratch bytes, not these archive
files. `.migration-v1/manifest.json` records the source inventory, descriptor/home
mapping, typed database checks and final file inventory. Its own file is excluded
from that inventory to avoid self-reference; containing-directory timestamps are
restored. Publication verifies the complete target, including the report bytes.

Publication reserves the target with exclusive `mkdir`, moves prepared entries,
and publishes `registry.json` last. It is not an atomic directory swap. Do not
start a service on the target until the command succeeds. Ordinary failures
remove only the target inode created by this attempt; a process or host crash
may leave an incomplete directory that requires inspection before retry. The
command neither overwrites an existing target nor offers crash-durable backup
guarantees. The new state root is deliberately private (`0700`).

## Startup Selection Preparation

`tools/startup_state.py` classifies all workspace databases, including archived
workspaces, through scratch copies. It rejects mixed, unknown or damaged formats.
This inspection does not establish that old writers have stopped.

`tools/startup_selection.py` provides the selection-record component for
managed cutover. Before first publication, it checks the original source
and migrated target against the successful migration report, the actual provider
and runner mappings, the migration mode and the target database format. The
record resides outside both state roots and external workspace repo/log paths.
Publication is exclusive and syncs the file and containing directory.

Subsequent loads bind the same source/target directory identities and immutable
migration report but permit normal new-version data changes. A changed or invalid
record fails; it never redirects silently to old data. Both original and target
directories must still exist while starting through this selection record.
After a directory-sync error, a record may already be visible: inspect/load it
instead of overwriting it or deleting the migrated data.

`tools/startup_migration.prepare_startup` coordinates these components and yields
the root to serve. Current state without a selection needs no conversion. Legacy
state requires a deployment-owned `quiesce(source)` context that stops all writers
and suppresses their restart throughout the new service's lifetime. The source
directory lock only serializes new coordinators; the application owns the target
lock. Completed but unpublished targets can be adopted after full verification;
partial targets and invalid selection records fail without replacement.

The quiescence context must never restart the old deployment on exit, including
when the new service fails: new writes may already exist. This callback is an
integration requirement, not proof of shutdown by itself.

`tools/tmux_deployment.py` provides a concrete owner for a dedicated legacy tmux
server with declared scripts, no external restart supervisor and no daemonized
host children. `capture_deployment` records source/socket/process identities,
exact panes, script hashes and source-mounted container identities for review.
`TmuxDeployment.quiesce` cancels the audited host sessions, verifies their children
have exited, then disables container restart and stops the audited containers.
Containers and writable layers are retained. New containers, changed identities,
foreign processes and active broader writable mounts fail instead of expanding ownership.
By default, unrelated broader mounts are ignored only when the container is explicitly stopped
and its restart policy is `no` or `unless-stopped`; they are never adopted as owned
containers. Unknown state or other restart policies fail the broader-mount check.
An operator may explicitly accept specified external containers remaining active
with `capture_deployment(..., external_containers=(full_id, ...))`. The report
records each complete ID, name, image and mount inventory; every shutdown check
revalidates them, ignoring mount order but preserving all fields and duplicates.
These exceptions can have broader mounts only, never direct source mounts, and
are never stopped by the adapter. Unknown broader writers still fail. This is
an explicit acceptance of possible external writes, not filesystem isolation;
source-change detection and independent backup verification remain necessary.
The owner must separately exclude independent host writers; manual restarts are
outside this dedicated deployment contract. No old service is restarted on exit.

An optional `TmuxDeployment(..., receipt=path, target=target)` persists completed
shutdown outside both state roots and external source repo/log paths. The exclusive
0600 record binds the frozen deployment report, target path and Linux boot ID.
Publication follows all shutdown checks and syncs the file and parent directory.
Callers serialize this operation through the startup coordinator's source lock.
After successful publication, later instances only verify the socket remains
inactive and the same containers remain stopped with restart disabled. They never
scan or signal the retired process IDs, which may have been reused. A damaged or
mismatched record, reboot, restarted service or changed container set requires
review; it never falls back to issuing shutdown commands. Publication failure
does not restart old services. The target selection record still independently
validates the migration and provider mappings; this receipt cannot replace it.

The managed entry below connects these components. These checks do not prove
provider resume or exclude independent writers. Use it for cutover, then validate
the published target with `selected-root` and adopt ordinary `serve` with that
target as `VIBESIM_AGENT_WORKSPACES_ROOT`. Do not make the old shutdown receipt,
container inventory or boot ID permanent normal-startup dependencies.

### Recovery After A Host Reboot

A shutdown receipt is bound to the Linux boot ID. After a reboot, managed
startup refuses that receipt even when migration was already completed. The
current capture tool requires a running legacy deployment; it cannot recertify
an already stopped deployment. Do not restart the old backend just to capture a
new receipt, and do not edit the recorded boot ID to bypass verification.

For an already published target, an operator can review an explicit transition
to ordinary current-state startup:

1. Keep admission closed and preserve both roots, selection, receipt and migration
   report. Confirm the old services, restart supervisors, containers and any
   independent writers are stopped and will remain stopped.
2. Validate the existing selection and its target using `selected-root` with the
   original startup configuration and provider configuration. This read-only
   command validates selection, source/target state and provider configuration;
   it does not renew the shutdown receipt. Stop for manual
   investigation if validation fails or migration was never published.
3. Set `VIBESIM_AGENT_WORKSPACES_ROOT` to that exact validated current-state
   target and start ordinary `serve` without `--startup-config`. Configure
   Analyzer to read the same target's `registry.json`. Ordinary startup still
   checks schemas and state ownership, but does not enforce legacy shutdown;
   the reviewed deployment configuration now owns that responsibility.
4. Verify historical reads, provider scope/session configuration and both
   services' selected paths before reopening admission. Preserve new writes;
   never replace the target with the old snapshot.

Perform this ordinary-startup transition as part of a successful cutover rather
than waiting for a reboot. Retain the original evidence and record the replacement
service configuration. Partial or unpublished migrations require separate
inspection and cannot use this path.

## Managed Startup Entry

`python -m vibesim_agent serve --startup-config /absolute/deployment/startup.json`
holds the shutdown/migration context through application creation, Uvicorn serving
and shutdown. This is the cutover entry, not the permanent production startup
command. Without this option, `serve` uses the initialized root configured by
`VIBESIM_AGENT_WORKSPACES_ROOT`. Application settings and runtime namespaces use
the selected root during cutover.
Failure never automatically restarts the legacy service or overwrites a target.

The JSON file has exactly these fields (all file paths must be absolute):

```json
{
  "format": 1,
  "source": "/deployment/legacy-state",
  "target": "/deployment/new-state",
  "selection": "/deployment/control/selection.json",
  "receipt": "/deployment/control/shutdown.json",
  "deployment": "/deployment/control/audited-tmux.json",
  "mapping": "/deployment/control/provider-mapping.json",
  "mode": "production"
}
```

The deployment file is the unwrapped result of `capture_deployment`; it is an
explicitly reviewed shutdown scope, not automatic discovery of permission.
The mapping uses the offline migration command's `models`, `families`, `runners`
and optional `external_paths` fields. `rehearsal` requires independent external
paths as described above. Control files must be distinct and outside both state
roots and all external workspace repositories and logs. Do not create the target
directory for service logs before migration; keep startup logs under the control
directory until a target has been successfully published.

Before shutdown, the entry validates the source checkout, mapped `w_main`, runtime
provider scopes, historical runner mappings and every old workspace database/home
mapping through scratch SQLite copies. Model settings without a historical session
do not require an invented session family. Shutdown still requires the adapter's
explicit local `DOCKER_HOST=unix:///var/run/docker.sock` environment and deployment
ownership assumptions. Its receipt prevents reusing retired numeric PIDs on later
starts. A host reboot currently requires a fresh deployment review.

Runtime provider scopes are checked again after migration, before handing the
selected environment to the application. If a profile changed during the copy,
startup fails while retaining the target and its selection; restore or explicitly
review the configuration rather than remapping existing sessions automatically.
This does not make concurrent profile editing transactional; adapter guards still
check profile identity at invocation time.

After Agent has published a target, the deployment owner starts Analyzer with
`--workspace-registry <selected-root>/registry.json`, obtaining the root from:

```bash
uv run python -m vibesim_agent selected-root --startup-config /absolute/deployment/startup.json
```

Run this read-only command with the same runtime/provider configuration as Agent,
including the same credential-discovery wrapper when used. It never stops services,
migrates data or guesses a target. Missing selection on legacy state, invalid
selection, changed configuration or changed external paths fails the command;
the caller must propagate failure instead of falling back to another registry.
It is a root resolver, not an Analyzer supervisor or proof that Agent is healthy.

After validating the result, persist the returned absolute target in the ordinary
Agent and Analyzer service configuration. Start Agent with `serve` and no
`--startup-config`; Analyzer reads that target's `registry.json`. Keep the selection
and receipt as migration evidence. Their validation does not replace HTTP health,
provider resume or coordinated service checks.

`verified=true` means the assembled state passed conversion and file checks.
`resume_verified`, `external_paths_verified` and `execution_isolation_verified`
remain false. Dependency entries report literal symlinks and Git `gitdir` pointers
with lexical paths at the final destination, not resolved link chains. Every
dependency has `requires_validation=true`; internal-looking links are not proof
of isolation. Unrecognized `.git` files are preserved and flagged for review,
including binary or oversized files. No deployment acceptance follows from the
conversion flag alone.

## Database Component

`tools/migrate_v1_database.py` converts one legacy v8 SQLite database to the
current `vibesim_agent` schema. It is a database component of the offline
workspace migration, not a complete workspace migration or deployment command.

The source uses `mode=ro`, `query_only` and one read transaction. Schema checks,
mapping, digests and copying share that snapshot, including committed WAL data.
A running writer can advance afterward; the target still contains the original
snapshot. This does not freeze role homes, repositories, logs or other databases.

## Mapping

The mapping JSON has two objects, `models` and `families`. Each maps an exact
source string to an object with `provider_id` and `session_scope`. Derive scope
from the target provider configuration, then validate historical CLI resume
compatibility separately. Do not infer an old session's provider from its
conversation's current model. An empty family can be mapped explicitly.

| Source | Target | Preservation |
| --- | --- | --- |
| `conversations` common columns | `conversations` | Original cell values |
| Three roles' model/effort/tier columns | `role_settings` | One row per role; values unchanged; provider/scope explicit |
| `codex_sessions` | `agent_sessions` | Conversation, role and session ID unchanged; family maps independently |
| Messages, turns, events, jobs, experiments and relations | Same tables | Every original cell, including nullable turn IDs and raw JSON |
| `sqlite_sequence` | `sqlite_sequence` | Original rows and values, including empty tables with retained high watermarks |
| `schema_migrations` | Returned manifest | Every version and application timestamp |

There is no model normalization, effort fallback, JSON reserialization, event
renumbering or automatic recovery during conversion. TEXT and BLOB are distinct
in the verification digest; NULL, integers and exact floating-point values also
have distinct encodings. The manifest hashes scope identities instead of
printing them and contains no conversation text.

## Commands

```bash
uv run python -m tools.migrate_v1_database source.sqlite --mapping profiles.json
uv run python -m tools.migrate_v1_database source.sqlite --mapping profiles.json --target converted.sqlite
```

Omitting `--target` performs preflight and computes expected mapped table digests
without output files; `verified` remains false. With a target, the converter
builds a private staged database, verifies content, integrity and foreign keys,
closes target connections, then publishes with an exclusive hard link. It never
replaces an existing target or source sidecar. Retain the returned manifest with
the migration artifacts.

Preflight checks table and column layout, types, nullability, primary keys,
unique indexes (including collation) and foreign keys. Unknown tables, columns,
generated columns, views and triggers fail. The schema digest records DDL;
this is not a general SQL migration engine for arbitrary custom CHECK clauses
or defaults. Version history must end at legacy v8. Cross-conversation links and
foreign-key defects fail. Historical `experiments.job_id` references to deleted
jobs are retained and reported, matching existing conversation deletion.

## File Copy Component

`tools.migration_files.inventory_tree(source)` inventories a real directory
without following symbolic links. `copy_tree(source, target)` copies into an
absent target, verifies its inventory, and rescans the source for changes.
The destination parent must already exist and remain controlled by the caller.
Keep the copy inside a private migration staging directory until all workspace
conversion and verification steps pass. A returned manifest is not a durable
publication or whole-workspace migration result.

The inventory includes every directory, regular file and symbolic link, including
dotfiles, untracked work, build outputs and logs. It records file SHA256, size,
mode and modification time; link targets retain their literal text. Hardlinks
are reconstructed only among destination files, never between source and target.
Special files such as sockets and FIFOs require explicit handling and fail the
copy. Logical and unique-inode byte counts support a free-space preflight; sparse
files may consume their full logical size, and the estimate does not include
filesystem metadata or concurrent disk usage.

This component does not preserve ownership, access/change times, sparse
allocation, ACLs or extended attributes. Source writers must already be stopped:
two matching scans cannot prove an instantaneous snapshot. Symbolic links and
Git worktree `gitdir` references may still point outside the copy; the enclosing
migration must report and resolve these dependencies before claiming an isolated
rehearsal. The manifest must be retained with the migration report.

## Workspace Assembly Rules

The enclosing converter must preserve descriptor identity, naming/state fields,
timestamps and base-workspace/revision values. Managed `repo` and `repo/logs`
remain relative to their workspace. External paths must first be resolved against
the original descriptor directory, then stored as absolute paths; copying a
relative external path verbatim would reinterpret it at the destination. Rebuild
the active `registry.json` from the converted descriptors and retain the original
index as migration evidence.

The production `w_main` keeps its original external repository and logs. A
rehearsal must explicitly substitute an independent writable repository and logs
copy. Database log-relative paths, resource IDs, peer-workspace values and raw
historical JSON remain unchanged. Historical absolute paths inside JSON are
evidence, not configuration to rewrite with global string replacement.

Preserve the complete original `codex/` tree, `jobs/` and unknown workspace files.
Archive the original database and sidecars away from the converted
`workspace.sqlite`; an old WAL must never sit beside the converted database.
Active provider homes use each stored session's own family mapping, independently
of current role model settings:

| Historical home | New active home |
| --- | --- |
| Codex `codex/<conversation>/<role>/` | `runtime/<conversation>/<role>/<sha256(scope)>/` |
| Claude `codex/<conversation>/<role>/claude/` | Same scope-derived root, with the child contents directly inside |

Use the runtime's `role_home` mapping so migration and container mounts agree.
A historical role directory can contain several providers' files after model
changes; refreshed auth/config files alone do not prove session ownership. Do
not activate homes without a stored session or explicit additional mapping.
Missing historical homes prevent a resume claim even if the database verifies.
Codex's earlier shared-only layout is supported: when the role directory is
absent but shared session directories exist, create an empty active scope home
and import them. Claude does not use this fallback. A session with neither a
role home nor a supported shared home fails preflight.

The old Codex shared-home fallback merges conversation-level `sessions/` and
`shell_snapshots/` when a role lacks `.legacy-shared-runtime-imported`. Preserve
both originals and require an explicit conflict policy when a shared and role
file at the same path have different contents. The current command rejects these
conflicts; equal existing file contents and link targets keep the role copy.
Original versions remain in the unchanged `codex/` tree. Active homes are derived
copies; hardlink groups crossing a shared/role merge are not guaranteed to stay
linked, while the original tree preserves those relationships. A migration must
not mark real resume validation complete from file presence alone.

## Real Resume Rehearsal

Prepare each external repository in its own bundle with
`tools/rehearsal_repository.py` after stopping its working-tree and shared Git
metadata writers:

```bash
uv run python -m tools.rehearsal_repository SOURCE_REPO NEW_BUNDLE --source-quiesced
```

Use `NEW_BUNDLE/repo` as the workspace override. The helper copies the whole
working tree, including staged, unstaged, untracked and ignored files. It replaces
each initialized repository's Git marker with an independent `.git` directory,
including linked worktrees and initialized nested submodules. It uses Git's
[local mirror clone with `--no-hardlinks`](https://git-scm.com/docs/git-clone),
preserves HEAD and refs, copies index state (expanding a split index in the copy),
and checks staged entries and working-file hashes. Original Git markers, indexes
and local rules are retained under `NEW_BUNDLE/evidence`; the manifest records
the checks. Local mirror cloning also requires stopped shared-metadata writers.

The derived Git configuration has no remotes and disables hooks and fsmonitor.
A bounded set of non-executable core options is retained: filemode, ignorecase,
autocrlf, eol, symlinks, precomposeunicode, trustctime and checkstat. Git parses
boolean values so an explicitly empty value does not become a default. Local
`info/exclude` and `info/attributes` are copied. Global/system Git configuration
and inherited `GIT_*` overrides are disabled during preparation. No worktree diff
or external filter is run; index entries and file content are checked separately.
This is not a backup of all source configuration, reflogs or in-progress Git
operations, and does not promise identical presentation under every Git setting.

Alternates, partial/promisor objects, sparse checkouts, unmerged indexes, active
merge/rebase/bisect operations, configured filters and external ignore/attribute
files require explicit handling and are rejected. Uninitialized gitlink folders
remain as copied files and are reported; they are not claimed to be initialized
submodules. Final Git checks execute inside the publication cleanup boundary,
so a verification failure removes only this attempt's target bundle.

`git_verified=true` covers the supported repositories' metadata isolation and
preserved Git/file state. `execution_isolation_verified` stays false. Resolve
symlinks and recreate checkout-bound environments before executing the copy.
Copied `.venv` and build directories preserve input evidence; their presence
does not certify that they can run independently of their original paths.

Fix the rehearsal state root, independent main repository, provider profile
locations, endpoints and CLI image versions before computing migration scopes.
Codex scope includes its configured profile home; relocating that profile later
changes compatibility. Set `VIBESIM_AGENT_MAIN_DIR` to the independent main copy
as well as overriding external descriptors. The service derives container
ownership from the resolved state root, so retain that location across restart
checks.

Check the workspace's per-user port convention before starting any service.
Configure the Agent port, container-reachable managed callback URL and bind
address together; a loopback-only listener is generally not reachable through
the container's host gateway. Set an explicit Analyzer endpoint for the rehearsal.
Inspect the resulting container mounts before sending a turn: writable repo and
role homes must refer only to rehearsal copies, while prompts, MCP, instructions,
submodules, peer repositories and configured model cache have the intended
read-only mounts. Resolve every reported external link and Git dependency before
claiming the repository is isolated.

For each historical role under test, first confirm that the new database's
provider and scope match the final provider configuration. Then capture evidence
that the CLI actually receives `resume` with the original session ID. A successful
new session is not a resume result. Send a minimal read-only turn, verify the new
turn/message and rollout files under the new root, and confirm that original
state and repositories remain unchanged. Restart the new service at the same
root and repeat. For orchestrated mode, verify orchestrator and implementer
sessions separately. Check targeted cancellation and a subsequent resume too.

When existing data lacks a required provider or role, create a baseline using the
legacy implementation in a separate fixture workspace and container. After it
creates real sessions, stop its writers, migrate the fixture and verify the same
session IDs with the current implementation. Keep the `/workspace` cwd and
CLI/backend identity consistent. Cover both historical roles before claiming
dual-role resume; single-role success and database/file conversion do not
establish that result. Complete the [cutover checks](cutover.md) separately.
