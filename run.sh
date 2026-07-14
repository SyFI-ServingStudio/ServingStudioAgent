#!/usr/bin/env bash
# Launch the VibeSim chat UI. Backend shells out to `codex`; it does not import VibeSim,
# so this runs in its own light uv venv (fastapi + uvicorn only).
set -euo pipefail
cd "$(dirname "$0")"

export CODEX_DOCKER_IMAGE="${CODEX_DOCKER_IMAGE:-vibesim-ui-codex-runner:latest}"
export CODEX_RUNNER_IMAGE_VERSION="${CODEX_RUNNER_IMAGE_VERSION:-prebuilt-codex-runner-v5}"
export CODEX_DOCKER_GPUS="${CODEX_DOCKER_GPUS-all}"
export CODEX_DOCKER_DG_USE_LOCAL_VERSION="${CODEX_DOCKER_DG_USE_LOCAL_VERSION:-0}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"

if [ "${FRONTEND_SKIP_BUILD:-0}" != "1" ]; then
  if [ ! -d frontend/node_modules ]; then
    (cd frontend && npm ci)
  fi
  (cd frontend && npm run build)
fi

if [ "${CODEX_SKIP_IMAGE_BUILD:-0}" != "1" ]; then
  image_version="$(docker image inspect -f '{{ index .Config.Labels "org.vibesim.ui.codex-runner.version" }}' "$CODEX_DOCKER_IMAGE" 2>/dev/null || true)"
  image_lock_sha="$(docker image inspect -f '{{ index .Config.Labels "org.vibesim.ui.main-lock-sha" }}' "$CODEX_DOCKER_IMAGE" 2>/dev/null || true)"
  current_lock_sha="$(sha256sum ../main/uv.lock | awk '{print $1}')"
  if [ "${CODEX_FORCE_IMAGE_BUILD:-0}" = "1" ] \
    || [ "$image_version" != "$CODEX_RUNNER_IMAGE_VERSION" ] \
    || [ "$image_lock_sha" != "$current_lock_sha" ]; then
    ./scripts/build-codex-runner-image.sh
  fi
fi

exec uv run uvicorn backend.app:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8765}" "$@"
