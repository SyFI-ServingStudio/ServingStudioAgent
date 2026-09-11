#!/usr/bin/env bash
# State initialization, migration and runner builds are explicit operations.
set -euo pipefail
cd "$(dirname "$0")"

exec uv run --frozen python -m vibesim_agent serve "$@"
