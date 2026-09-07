#!/usr/bin/env bash
# Post-build acceptance test for the Codex runner image. This intentionally runs
# as the production non-root identity against an isolated tracked VibeSim copy;
# checking only that tool binaries exist misses ownership and linker failures.
set -euo pipefail

ui_dir="$(cd "$(dirname "$0")/.." && pwd)"
workspace_dir="$(cd "$ui_dir/.." && pwd)"
main_dir="$workspace_dir/main"

default_image_owner="$(id -un | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9_.-' '-')"
default_image_owner="${default_image_owner%-}"
image="${CODEX_DOCKER_IMAGE:-vibesim-ui-codex-runner:${default_image_owner:-codex}}"
app_uid="${CODEX_DOCKER_UID:-$(id -u)}"
app_gid="${CODEX_DOCKER_GID:-$(id -g)}"
app_user="${CODEX_DOCKER_USER:-${USER:-kanzhu}}"
app_home="${CODEX_DOCKER_HOME:-/home/$app_user}"
smoke_level="${1:-build}"

# shellcheck source=lib/main-tree-copy.sh
source "$ui_dir/scripts/lib/main-tree-copy.sh"

case "$smoke_level" in
  build|timing) ;;
  *)
    echo "usage: $0 [build|timing]" >&2
    exit 2
    ;;
esac

if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "runner image does not exist: $image" >&2
  exit 2
fi

smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/vibesim-runner-smoke.XXXXXX")"
smoke_workspace="$smoke_root/main"
cleanup() {
  rm -rf "$smoke_root"
}
trap cleanup EXIT

# Assembled by the same helper the image build context uses, so this really
# tests the source the image was built from. See scripts/lib/main-tree-copy.sh.
copy_main_tree "$main_dir" "$smoke_workspace"
git -C "$smoke_workspace" init -q
git -C "$smoke_workspace" add -A
git -C "$smoke_workspace" \
  -c user.name=vibesim-runner-smoke \
  -c user.email=vibesim-runner-smoke@localhost \
  commit -qm "runner smoke workspace"

docker_args=(
  run --rm
  --user "$app_uid:$app_gid"
  --workdir /workspace
  --volume "$smoke_workspace:/workspace"
  --env "HOME=$app_home"
  --env DG_USE_LOCAL_VERSION=0
  "$image"
)

if [ "$smoke_level" = timing ]; then
  docker_args=(
    run --rm
    --gpus "${CODEX_DOCKER_GPUS:-all}"
    --user "$app_uid:$app_gid"
    --workdir /workspace
    --volume "$smoke_workspace:/workspace"
    --env "HOME=$app_home"
    --env DG_USE_LOCAL_VERSION=0
    "$image"
  )
fi

docker "${docker_args[@]}" bash -lc '
set -euo pipefail

fail() {
  echo "runner image smoke failed: $*" >&2
  exit 1
}

require_writable_env_dir() {
  env_name="$1"
  env_value="${!env_name:-}"
  [ -n "$env_value" ] || fail "$env_name is unset"
  [ -d "$env_value" ] || fail "$env_name is not a directory: $env_value"
  [ -w "$env_value" ] || fail "$env_name is not writable by $(id -u):$(id -g): $env_value"
}

[ "$(id -u)" = "'"$app_uid"'" ] || fail "unexpected UID $(id -u), expected '"$app_uid"'"
[ "$(id -g)" = "'"$app_gid"'" ] || fail "unexpected GID $(id -g), expected '"$app_gid"'"
[ -d "$HOME" ] || fail "HOME is not a directory: $HOME"
[ -w "$HOME" ] || fail "HOME is not writable: $HOME"
require_writable_env_dir CARGO_HOME
require_writable_env_dir UV_PROJECT_ENVIRONMENT
require_writable_env_dir UV_CACHE_DIR

for tool in cargo cc claude codex ld mold protoc python rustc uv; do
  command -v "$tool" >/dev/null || fail "required tool is missing: $tool"
done

# Exercise the linker selected by main/.cargo/config.toml before the expensive
# workspace build so a broken image fails with a small, local diagnostic.
native_smoke_dir="$(mktemp -d /workspace/.runner-native-smoke.XXXXXX)"
trap '\''rm -rf "$native_smoke_dir"'\'' EXIT
printf '\''int main(void) { return 0; }\n'\'' > "$native_smoke_dir/main.c"
cc -fuse-ld=mold "$native_smoke_dir/main.c" -o "$native_smoke_dir/main"
"$native_smoke_dir/main"

# Seed exactly as container startup does, then exercise the same launcher build
# environment used by an agent. The image intentionally excludes first-party
# VibeSim artifacts so any workspace revision is safe; only those local packages
# may compile here. Recompiling a third-party crate means the dependency seed
# drifted or was incomplete.
test -d "${VIBESIM_BAKED_TARGET:?}/release" || fail "Cargo target seed is missing"
test ! -e "${VIBESIM_BAKED_TARGET}/release/simulator" \
  || fail "Cargo target seed contains a stale simulator binary"
test ! -e "${VIBESIM_BAKED_TARGET}/release/analyze" \
  || fail "Cargo target seed contains a stale analyzer binary"
mkdir -p target
cp -a --reflink=auto "$VIBESIM_BAKED_TARGET/." target/
launcher_log="$(mktemp /workspace/.runner-launcher-smoke.XXXXXX.log)"
uv run python -m launcher --cache-report presets/unified_smoke.yaml 2>&1 | tee "$launcher_log"
if grep -E "^[[:space:]]*Compiling " "$launcher_log" \
  | grep -Ev "^[[:space:]]*Compiling (simulator|analyzer|timing-kernel-derive|schema-derive) "; then
  fail "Cargo target seed caused a third-party dependency to recompile"
fi

if [ "'"$smoke_level"'" = timing ]; then
  nvidia-smi -L >/dev/null
  uv run python -m launcher timing-predict presets/predict_llama3_8b_iter.json
  test -s logs/predict_llama3_8b_iter/reports/iter_breakdown.ans \
    || fail "timing-predict did not produce reports/iter_breakdown.ans"
fi
'

echo "runner image smoke passed: image=$image level=$smoke_level uid=$app_uid gid=$app_gid"
