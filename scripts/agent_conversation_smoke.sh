#!/usr/bin/env bash
# End-to-end smoke for the VibeSim agent CONVERSATION API (the real interactive
# interface): create -> turn -> follow-up turn -> history -> delete. When
# VIBESIM_API_TOKEN is set it also asserts a tokenless create is rejected with 401.
#
# Requires the backend running (./run.sh) and, for the turn steps, Docker + Codex
# auth (each turn spins up / reuses the conversation's isolated container). The
# two turns use a read-only prompt so the smoke stays cheap. Needs curl + python3.
#
# Usage:
#   scripts/agent_conversation_smoke.sh [BASE_URL]
#   VIBESIM_API_TOKEN=secret scripts/agent_conversation_smoke.sh http://127.0.0.1:8765
set -euo pipefail

BASE="${1:-${VIBESIM_BASE_URL:-http://127.0.0.1:8765}}"
TOKEN="${VIBESIM_API_TOKEN:-}"
PROMPT1="${SMOKE_PROMPT1:-Which VibeSim L1 profilers are available? Do not change any files.}"
PROMPT2="${SMOKE_PROMPT2:-Thanks. Of those, which one would cost a bf16 GEMM?}"

AUTH=()
if [ -n "$TOKEN" ]; then
  AUTH=(-H "Authorization: Bearer $TOKEN")
fi

json_get() { python3 -c 'import sys,json; d=json.load(sys.stdin); print(d'"$1"')'; }
post_body() { python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1])))' "$1"; }

echo "== 0. GET /api/agent/skill (public) =="
curl -fsS "$BASE/api/agent/skill" | grep -q "VibeSim" && echo "  ok: skill doc served"

if [ -n "$TOKEN" ]; then
  echo "== 0b. POST /api/agent/conversations without token -> expect 401 =="
  code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/api/agent/conversations" \
    -H 'Content-Type: application/json' -d '{}')"
  [ "$code" = "401" ] && echo "  ok: rejected ($code)" || { echo "  FAIL: got $code"; exit 1; }
fi

echo "== 1. create conversation =="
conv="$(curl -fsS -X POST "$BASE/api/agent/conversations" "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d "$(post_body '{"sandbox":"read-only","autonomous":false}')")"
cid="$(echo "$conv" | json_get '["id"]')"
echo "  cid=$cid"
[ -n "$cid" ] || { echo "  FAIL: no id"; exit 1; }

turn() {  # $1 = prompt text
  local body; body="$(python3 -c 'import json,sys; print(json.dumps({"text":sys.argv[1]}))' "$1")"
  curl -fsS -X POST "$BASE/api/agent/conversations/$cid/messages" "${AUTH[@]}" \
    -H 'Content-Type: application/json' -d "$body"
}

echo "== 2. first turn (may take a while) =="
r1="$(turn "$PROMPT1")"
echo "  ok=$(echo "$r1" | json_get '["ok"]') final_len=$(echo "$r1" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["final"]))')"
echo "  sessions=$(echo "$r1" | python3 -c 'import sys,json;print(sorted(json.load(sys.stdin)["sessions"]))')"

echo "== 3. follow-up turn (same conversation, resumed session) =="
r2="$(turn "$PROMPT2")"
echo "  ok=$(echo "$r2" | json_get '["ok"]') final_len=$(echo "$r2" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["final"]))')"

echo "== 4. history =="
hist="$(curl -fsS "${AUTH[@]}" "$BASE/api/agent/conversations/$cid")"
echo "  messages=$(echo "$hist" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["messages"]))')  (expect 4: 2 user + 2 assistant)"

echo "== 5. delete =="
curl -fsS -X DELETE "${AUTH[@]}" "$BASE/api/agent/conversations/$cid" >/dev/null && echo "  deleted"

echo "== conversation smoke OK =="
