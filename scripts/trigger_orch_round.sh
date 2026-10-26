#!/usr/bin/env bash
# Optional WORKFLOW_TRIGGER_SCRIPT: after tracer seal, HTTP-kick whoever exposes
# POST /api/orchestrator/run (typically the RSI dashboard). Tracer does not import
# orch or agent code — PANEL_URL is a pure HTTP contract.
set -euo pipefail

PANEL_URL="${PANEL_URL:-http://127.0.0.1:8165}"
PANEL_URL="${PANEL_URL%/}"
ACCESS_KEY="${DASHBOARD_ACCESS_KEY:-${AVI_DASHBOARD_ACCESS_KEY:-}}"

hdrs=(-H "Content-Type: application/json")
if [[ -n "${ACCESS_KEY}" ]]; then
  token="$(
    python3 - <<'PY' "${ACCESS_KEY}"
import hashlib, hmac, sys
print(hmac.new(sys.argv[1].encode(), b"rsi-dashboard-auth-v1", hashlib.sha256).hexdigest())
PY
  )"
  hdrs+=(-H "Cookie: rsi_dash_auth=${token}")
fi

# Non-blocking kick: /run may hold until the round finishes on the server side.
curl -sS -m 15 -X POST "${PANEL_URL}/api/orchestrator/run" \
  "${hdrs[@]}" \
  -d '{}' \
  -o /dev/null -w "trigger_orch_round HTTP %{http_code}\n" \
  || true
