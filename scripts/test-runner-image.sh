#!/usr/bin/env bash
# Exercise the runner as its configured identity against a private tracked copy.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
smoke_level=build
main_override=""
if [ "$#" -gt 0 ] && [ "$1" != "--main-dir" ]; then
  smoke_level="$1"
  shift
fi
if [ "$#" -gt 0 ]; then
  if [ "$#" -ne 2 ] || [ "$1" != "--main-dir" ]; then
    echo "usage: $0 [build|timing] [--main-dir PATH]" >&2
    exit 2
  fi
  main_override="$2"
fi
case "$smoke_level" in
  build|timing) ;;
  *) echo "usage: $0 [build|timing] [--main-dir PATH]" >&2; exit 2 ;;
esac

smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/vibesim-runner-smoke.XXXXXX")"
cleanup() {
  smoke_status=$?
  if [ "$smoke_status" -ne 0 ] && [ -d "$smoke_root/main/logs" ]; then
    echo "runner image smoke failed; private workspace and diagnostics retained: $smoke_root" >&2
  else
    rm -rf "$smoke_root"
  fi
  return "$smoke_status"
}
trap cleanup EXIT
# A regular file propagates configuration failures, unlike process substitution.
uv run --frozen python - "$repo_root" "$main_override" > "$smoke_root/config" <<'PY'
import os
import sys
from pathlib import Path
from vibesim_agent.bootstrap import configuration
from vibesim_agent.settings import ConfigurationError

environment = dict(os.environ)
if sys.argv[2]:
    environment['VIBESIM_AGENT_MAIN_DIR'] = str(Path(sys.argv[2]).resolve())
try:
    settings = configuration(environment=environment, repo_root=Path(sys.argv[1]))
except ConfigurationError as error:
    raise SystemExit(str(error)) from None
c = settings.container
for value in (settings.agent.main_dir, c.image, c.uid, c.gid, c.user, c.home, c.gpus):
    sys.stdout.buffer.write(str(value).encode() + b'\0')
PY
mapfile -d '' -t config < "$smoke_root/config"
[ "${#config[@]}" -eq 7 ] || { echo "invalid runner configuration output" >&2; exit 2; }
main_dir="${config[0]}"
image="${config[1]}"
app_uid="${config[2]}"
app_gid="${config[3]}"
app_user="${config[4]}"
app_home="${config[5]}"
gpus="${config[6]}"
if [ "$smoke_level" = timing ] && [ -z "$gpus" ]; then
  echo "timing smoke requires nonempty VIBESIM_RUNNER_GPUS" >&2
  exit 2
fi

# Source copying and private Git initialization must not inherit a caller's index.
for name in ${!GIT_@}; do unset "$name"; done
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
# shellcheck source=lib/main-tree-copy.sh
source "$repo_root/scripts/lib/main-tree-copy.sh"
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "runner image does not exist: $image" >&2
  exit 2
fi
smoke_workspace="$smoke_root/main"
copy_main_tree "$main_dir" "$smoke_workspace"
git -C "$smoke_workspace" init -q
git -C "$smoke_workspace" add -A
git -C "$smoke_workspace" -c user.name=vibesim-runner-smoke \
  -c user.email=vibesim-runner-smoke@localhost commit -qm "runner smoke workspace"

docker_args=(run --rm -i --user "$app_uid:$app_gid" --workdir /workspace
  --volume "$smoke_workspace:/workspace" --env "HOME=$app_home"
  --env "USER=$app_user" --env "LOGNAME=$app_user" --env DG_USE_LOCAL_VERSION=0)
if [ "$smoke_level" = timing ]; then
  docker_args+=(--gpus "$gpus"
    --volume "$repo_root/scripts/lib/runner-timing-smoke.py:/opt/vibesim/runner-timing-smoke.py:ro")
fi
docker_args+=("$image")
docker "${docker_args[@]}" bash -s -- "$app_uid" "$app_gid" "$smoke_level" <<'SH'
set -euo pipefail
fail() { echo "runner image smoke failed: $*" >&2; exit 1; }
require_writable_env_dir() {
  env_name="$1"
  env_value="${!env_name:-}"
  [ -n "$env_value" ] || fail "$env_name is unset"
  [ -d "$env_value" ] || fail "$env_name is not a directory: $env_value"
  [ -w "$env_value" ] || fail "$env_name is not writable by $(id -u):$(id -g): $env_value"
}
[ "$(id -u)" = "$1" ] || fail "unexpected UID"
[ "$(id -g)" = "$2" ] || fail "unexpected GID"
[ -d "$HOME" ] || fail "HOME is not a directory: $HOME"
[ -w "$HOME" ] || fail "HOME is not writable: $HOME"
require_writable_env_dir CARGO_HOME
require_writable_env_dir UV_PROJECT_ENVIRONMENT
require_writable_env_dir UV_CACHE_DIR
for tool in cargo cc claude codex ld mold protoc python rustc uv; do
  command -v "$tool" >/dev/null || fail "required tool is missing: $tool"
done

native_smoke_dir="$(mktemp -d /workspace/.runner-native-smoke.XXXXXX)"
trap 'rm -rf "$native_smoke_dir"' EXIT
printf 'int main(void) { return 0; }\n' > "$native_smoke_dir/main.c"
cc -fuse-ld=mold "$native_smoke_dir/main.c" -o "$native_smoke_dir/main"
"$native_smoke_dir/main"

# Dependency seeds exclude first-party binaries, which must build from this copy.
test -d "${VIBESIM_BAKED_TARGET:?}/release" || fail "Cargo target seed is missing"
test ! -e "$VIBESIM_BAKED_TARGET/release/simulator" || fail "Cargo target seed contains a stale simulator binary"
test ! -e "$VIBESIM_BAKED_TARGET/release/analyze" || fail "Cargo target seed contains a stale analyzer binary"
mkdir -p target
cp -a --reflink=auto "$VIBESIM_BAKED_TARGET/." target/
launcher_log="$(mktemp /workspace/.runner-launcher-smoke.XXXXXX.log)"
uv run python -m launcher --cache-report presets/unified_smoke.yaml 2>&1 | tee "$launcher_log"
if grep -E "^[[:space:]]*Compiling " "$launcher_log" \
  | grep -Ev "^[[:space:]]*Compiling (simulator|analyzer|timing-kernel-derive|schema-derive) "; then
  fail "Cargo target seed caused a third-party dependency to recompile"
fi
if [ "$3" = timing ]; then
  uv run python /opt/vibesim/runner-timing-smoke.py
fi
SH
echo "runner image smoke passed: image=$image level=$smoke_level uid=$app_uid gid=$app_gid"
