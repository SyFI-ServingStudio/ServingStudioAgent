#!/usr/bin/env bash
# End-to-end smoke for the VibeSim agent HTTP API.
#
# Exercises: GET /api/agent/skill, POST /api/eval (read-only prompt), and workspace-scoped
# artifact list/download. When VIBESIM_API_TOKEN is set it also asserts that a
# tokenless /api/eval is rejected with 401.
#
# Requires the backend to be running (./run.sh) and, for the eval step, Docker +
# Codex auth (the eval spins up the isolated container). Needs curl + uv.
#
# Usage:
#   scripts/agent_api_smoke.sh [BASE_URL]
#   VIBESIM_API_TOKEN=secret scripts/agent_api_smoke.sh http://127.0.0.1:8765
set -euo pipefail

BASE="${1:-${VIBESIM_BASE_URL:-http://127.0.0.1:8765}}"
TOKEN="${VIBESIM_API_TOKEN:-}"
PROMPT="${SMOKE_PROMPT:-List the available VibeSim L1 profilers.}"

AUTH=()
if [ -n "$TOKEN" ]; then
  AUTH=(-H "Authorization: Bearer $TOKEN")
fi

# Extract one field from a JSON blob on stdin.
json_get() { uv run python -c 'import sys,json; d=json.load(sys.stdin); print(d'"$1"')'; }

echo "== 1. GET /api/agent/skill (public) =="
skill="$(curl -fsS "$BASE/api/agent/skill")"
echo "$skill" | grep -q "VibeSim Agent API" && echo "  ok: skill doc served"

if [ -n "$TOKEN" ]; then
  echo "== 1b. POST /api/eval without token -> expect 401 =="
  code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/api/eval" \
    -H 'Content-Type: application/json' -d '{"prompt":"ping"}')"
  [ "$code" = "401" ] && echo "  ok: rejected ($code)" || { echo "  FAIL: got $code"; exit 1; }
fi

echo "== 2. POST /api/eval (sandbox=read-only) — may take minutes =="
body="$(uv run python -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"sandbox":"read-only"}))' "$PROMPT")"
resp="$(curl -fsS -X POST "$BASE/api/eval" "${AUTH[@]}" \
  -H 'Content-Type: application/json' -d "$body")"
cid="$(echo "$resp" | json_get '["conversation_id"]')"
workspace_id="$(echo "$resp" | json_get '["workspace_id"]')"
ok="$(echo "$resp" | json_get '["ok"]')"
echo "  workspace_id=$workspace_id conversation_id=$cid ok=$ok"
[ -n "$cid" ] || { echo "  FAIL: no conversation_id"; exit 1; }
[ -n "$workspace_id" ] || { echo "  FAIL: no workspace_id"; exit 1; }

echo "== 3. GET workspace artifacts =="
listing="$(curl -fsS -G "$BASE/api/agent/workspaces/$workspace_id/artifacts" "${AUTH[@]}")"
count="$(echo "$listing" | json_get '["count"]')"
echo "  artifact count=$count"
first="$(echo "$listing" | uv run python -c 'import sys,json; f=json.load(sys.stdin)["files"]; print(f[0]["path"] if f else "")')"

if [ -n "$first" ]; then
  echo "== 4. GET workspace artifact download path=$first =="
  out="$(mktemp "$TMPDIR/vibesim-agent-artifact.XXXXXX")"
  curl -fsS -G "$BASE/api/agent/workspaces/$workspace_id/artifacts/download" "${AUTH[@]}" \
    --data-urlencode "path=$first" -o "$out"
  echo "  downloaded $(wc -c <"$out") bytes -> $out"
else
  echo "  (no files listed; skipping download)"
fi

echo "== smoke OK =="
