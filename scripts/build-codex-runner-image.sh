#!/usr/bin/env bash
set -euo pipefail

ui_dir="$(cd "$(dirname "$0")/.." && pwd)"
workspace_dir="$(cd "$ui_dir/.." && pwd)"
main_dir="$workspace_dir/main"
cd "$ui_dir"

default_image_owner="$(id -un | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9_.-' '-')"
default_image_owner="${default_image_owner%-}"
image="${CODEX_DOCKER_IMAGE:-vibesim-ui-codex-runner:${default_image_owner:-codex}}"
cuda_image="${CODEX_CUDA_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu24.04}"
uv_image="${CODEX_UV_IMAGE:-ghcr.io/astral-sh/uv:python3.12-bookworm}"
codex_package="${CODEX_NPM_PACKAGE:-@openai/codex@0.144.0}"
node_version="${NODE_VERSION:-v20.18.1}"
node_arch="${NODE_ARCH:-linux-x64}"
app_uid="${CODEX_DOCKER_UID:-$(id -u)}"
app_gid="${CODEX_DOCKER_GID:-$(id -g)}"
app_user="${CODEX_DOCKER_USER:-${USER:-kanzhu}}"
rust_toolchain="${RUST_TOOLCHAIN:-stable}"
runner_version="${CODEX_RUNNER_IMAGE_VERSION:-prebuilt-codex-runner-v10}"
lock_sha="$(sha256sum "$main_dir/uv.lock" | awk '{print $1}')"
build_context="$(mktemp -d "${TMPDIR:-/tmp}/vibesim-ui-runner-build.XXXXXX")"
# shellcheck source=lib/main-tree-copy.sh
source "$ui_dir/scripts/lib/main-tree-copy.sh"
trap 'rm -rf "$build_context"' EXIT

copy_main_tree "$main_dir" "$build_context/vibesim"

docker build \
  -f "$ui_dir/docker/codex-runner.Dockerfile" \
  -t "$image" \
  --build-arg "CUDA_IMAGE=$cuda_image" \
  --build-arg "UV_IMAGE=$uv_image" \
  --build-arg "CODEX_NPM_PACKAGE=$codex_package" \
  --build-arg "NODE_VERSION=$node_version" \
  --build-arg "NODE_ARCH=$node_arch" \
  --build-arg "APP_UID=$app_uid" \
  --build-arg "APP_GID=$app_gid" \
  --build-arg "APP_USER=$app_user" \
  --build-arg "RUST_TOOLCHAIN=$rust_toolchain" \
  --build-arg "RUNNER_VERSION=$runner_version" \
  --build-arg "VIBESIM_LOCK_SHA=$lock_sha" \
  "$build_context"

if [ "${CODEX_SKIP_RUNNER_IMAGE_TEST:-0}" != "1" ]; then
  "$ui_dir/scripts/test-codex-runner-image.sh" build
fi
