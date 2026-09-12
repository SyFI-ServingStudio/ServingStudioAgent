#!/usr/bin/env bash
# End-to-end smoke for the ServingStudioSim agent HTTP API.
#
# Exercises: GET /api/agent/v1/tools/skill, POST /api/agent/v1/tools/eval (read-only prompt), and workspace-scoped
# artifact list/download. When an API token is set it also asserts that a
# tokenless /api/agent/v1/tools/eval is rejected with 401.
#
# Requires the configured backend to be running (see README.md) and, for the
# eval step, Docker + provider auth. Needs curl + uv.
# VIBESIM_AGENT_API_TOKEN takes precedence, including an explicitly empty value;
# VIBESIM_API_TOKEN remains a fallback for the legacy backend.
#
# Usage:
#   scripts/agent_api_smoke.sh [BASE_URL]
#   VIBESIM_AGENT_API_TOKEN=secret scripts/agent_api_smoke.sh http://127.0.0.1:8765
set -euo pipefail

BASE="${1:-${VIBESIM_BASE_URL:-http://127.0.0.1:8765}}"
TOKEN="${VIBESIM_AGENT_API_TOKEN-${VIBESIM_API_TOKEN:-}}"
PROMPT="${SMOKE_PROMPT:-List the available ServingStudioSim L1 profilers.}"

AUTH=()
if [ -n "$TOKEN" ]; then
  AUTH=(-H "Authorization: Bearer $TOKEN")
fi

# Extract one field from a JSON blob on stdin.
json_get() { uv run python -c 'import sys,json; d=json.load(sys.stdin); print(d'"$1"')'; }

echo "== 1. GET /api/agent/v1/tools/skill (public) =="
skill="$(curl -fsS "$BASE/api/agent/v1/tools/skill")"
echo "$skill" | grep -q "ServingStudio Agent API" && echo "  ok: skill doc served"

if [ -n "$TOKEN" ]; then
  echo "== 1b. POST /api/agent/v1/tools/eval without token -> expect 401 =="
  code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/api/agent/v1/tools/eval" \
    -H 'Content-Type: application/json' -d '{"prompt":"ping"}')"
  [ "$code" = "401" ] && echo "  ok: rejected ($code)" || { echo "  FAIL: got $code"; exit 1; }
fi

echo "== 2. POST /api/agent/v1/tools/eval (sandbox=read-only) — may take minutes =="
body="$(uv run python -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"sandbox":"read-only"}))' "$PROMPT")"
resp="$(curl -fsS -X POST "$BASE/api/agent/v1/tools/eval" "${AUTH[@]}" \
  -H 'Content-Type: application/json' -d "$body")"
cid="$(echo "$resp" | json_get '["conversation_id"]')"
workspace_id="$(echo "$resp" | json_get '["workspace_id"]')"
ok="$(echo "$resp" | json_get '["ok"]')"
echo "  workspace_id=$workspace_id conversation_id=$cid ok=$ok"
[ -n "$cid" ] || { echo "  FAIL: no conversation_id"; exit 1; }
[ -n "$workspace_id" ] || { echo "  FAIL: no workspace_id"; exit 1; }

echo "== 3. GET workspace artifacts =="
listing="$(curl -fsS -G "$BASE/api/agent/v1/tools/workspaces/$workspace_id/artifacts" "${AUTH[@]}")"
count="$(echo "$listing" | json_get '["count"]')"
echo "  artifact count=$count"
first="$(echo "$listing" | uv run python -c 'import sys,json; f=json.load(sys.stdin)["files"]; print(f[0]["path"] if f else "")')"

if [ -n "$first" ]; then
  echo "== 4. GET workspace artifact download path=$first =="
  out="$(mktemp "$TMPDIR/vibesim-agent-artifact.XXXXXX")"
  curl -fsS -G "$BASE/api/agent/v1/tools/workspaces/$workspace_id/artifacts/download" "${AUTH[@]}" \
    --data-urlencode "path=$first" -o "$out"
  echo "  downloaded $(wc -c <"$out") bytes -> $out"
else
  echo "  (no files listed; skipping download)"
fi

echo "== smoke OK =="
