#!/usr/bin/env bash
# WORKFLOW_TRIGGER_SCRIPT for dual-slot pipeline: after agent flush+seal, ask
# the orchestrator to train the oldest ready batch (batch_id auto-picked).
set -euo pipefail

PANEL_URL="${PANEL_URL:-http://127.0.0.1:8165}"
PANEL_URL="${PANEL_URL%/}"
ACCESS_KEY="${DASHBOARD_ACCESS_KEY:-${AVI_DASHBOARD_ACCESS_KEY:-}}"

hdrs=(-H "Content-Type: application/json")
if [[ -n "${ACCESS_KEY}" ]]; then
  # Same cookie derivation as scripts/agent_batch_feeder.py / agent_data_loop.py
  token="$(
    python3 - <<'PY' "${ACCESS_KEY}"
import hashlib, hmac, sys
print(hmac.new(sys.argv[1].encode(), b"rsi-dashboard-auth-v1", hashlib.sha256).hexdigest())
PY
  )"
  hdrs+=(-H "Cookie: rsi_dash_auth=${token}")
fi

# Non-blocking kick: orch /run holds until the round finishes.
curl -sS -m 15 -X POST "${PANEL_URL}/api/orchestrator/run" \
  "${hdrs[@]}" \
  -d '{}' \
  -o /dev/null -w "trigger_orch_round HTTP %{http_code}\n" \
  || true
