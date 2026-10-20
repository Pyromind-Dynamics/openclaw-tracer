#!/usr/bin/env bash
# Smoke-test openclaw-tracer callbacks with echo mocks.
#
# Phase A (always): schedule workflow / routing / data-delete callbacks in-process.
# Phase B (optional): if SMOKE_BASE_URL is set, hit live APIs:
#   POST /flush?trigger_workflow=true
#   POST /admin/routing/deployments (needs ADMIN_API_KEY)
#
# Usage:
#   ./scripts/smoke-callbacks.sh
#   SMOKE_BASE_URL=http://127.0.0.1:43886 PROXY_API_KEY=... ADMIN_API_KEY=... \
#     ./scripts/smoke-callbacks.sh
#
# Env:
#   SMOKE_BASE_URL   optional live tracer URL
#   PROXY_API_KEY    for /flush (default demo-openclaw-proxy-key)
#   ADMIN_API_KEY    for /admin/routing (optional; skips admin smoke if unset)
#   SMOKE_KEEP_LOGS=1  keep mock log dir after success

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOCK_DIR="${ROOT_DIR}/scripts/smoke_callbacks_mocks"
LOG_DIR="${SMOKE_LOG_DIR:-$(mktemp -d /tmp/openclaw-smoke-cb-XXXXXX)}"
PROXY_API_KEY="${PROXY_API_KEY:-demo-openclaw-proxy-key}"
ADMIN_API_KEY="${ADMIN_API_KEY:-}"
SMOKE_BASE_URL="${SMOKE_BASE_URL:-}"

die() { echo "✗ $*" >&2; exit 1; }
info() { echo "==> $*"; }
ok() { echo "✓ $*"; }

[[ -d "${MOCK_DIR}" ]] || die "missing mocks: ${MOCK_DIR}"
chmod +x "${MOCK_DIR}"/*.sh

mkdir -p "${LOG_DIR}"
WF_LOG="${LOG_DIR}/workflow.log"
RT_LOG="${LOG_DIR}/routing.log"
DD_LOG="${LOG_DIR}/data_delete.log"

# Wrap mocks so stdout lands in log files (subprocess from Python has no TTY).
cat >"${LOG_DIR}/run_workflow.sh" <<EOF
#!/bin/bash
exec >>"${WF_LOG}" 2>&1
exec /bin/bash "${MOCK_DIR}/mock_workflow.sh" "\$@"
EOF
cat >"${LOG_DIR}/run_routing.sh" <<EOF
#!/bin/bash
exec >>"${RT_LOG}" 2>&1
exec /bin/bash "${MOCK_DIR}/mock_routing_callback.sh" "\$@"
EOF
cat >"${LOG_DIR}/run_data_delete.sh" <<EOF
#!/bin/bash
exec >>"${DD_LOG}" 2>&1
exec /bin/bash "${MOCK_DIR}/mock_data_delete_callback.sh" "\$@"
EOF
chmod +x "${LOG_DIR}"/run_*.sh

export WORKFLOW_TRIGGER_SCRIPT="${LOG_DIR}/run_workflow.sh"
export ENABLE_WORKFLOW_TRIGGER=1
export ROUTING_CHANGE_CALLBACK_SCRIPT="${LOG_DIR}/run_routing.sh"
export ROUTING_CHANGE_CALLBACK_URL=""
export DATA_DELETE_CALLBACK_SCRIPT="${LOG_DIR}/run_data_delete.sh"
export DATA_DELETE_CALLBACK_URL=""

info "mock logs → ${LOG_DIR}"

# ---------- Phase A: in-process schedule ----------
info "Phase A: schedule callbacks via Python"
cd "${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
import asyncio
import os
import time
from pathlib import Path

from openclaw_tracer.workflow_trigger import schedule_workflow_trigger, run_workflow_trigger
from openclaw_tracer.routing_callback import (
    build_routing_event,
    schedule_routing_change_callback,
)
from openclaw_tracer.data_callback import (
    build_data_delete_event,
    schedule_data_delete_callback,
)


async def main() -> None:
    wf = Path(os.environ["WORKFLOW_TRIGGER_SCRIPT"])
    # Direct await so smoke does not depend on create_task timing alone
    await run_workflow_trigger(wf)
    schedule_workflow_trigger(wf)

    schedule_routing_change_callback(
        build_routing_event(
            reason="smoke",
            previous=[{"model_name": "m", "litellm_params": {"api_key": "sk-secret"}}],
            current=[{"model_name": "m", "litellm_params": {"api_base": "http://b"}}],
            op={"method": "PATCH", "path": "/admin/routing/weights"},
        )
    )
    schedule_data_delete_callback(
        build_data_delete_event(
            reason="batch_advance",
            paths=["/tmp/smoke/spans/0"],
            batch_index=0,
            extra={"current_batch_index": 1},
        )
    )
    await asyncio.sleep(1.5)


asyncio.run(main())
print("phase A schedules done")
PY

# Wait a bit more for fire-and-forget tasks
sleep 1

[[ -s "${WF_LOG}" ]] || die "workflow mock log empty: ${WF_LOG}"
[[ -s "${RT_LOG}" ]] || die "routing mock log empty: ${RT_LOG}"
[[ -s "${DD_LOG}" ]] || die "data-delete mock log empty: ${DD_LOG}"
grep -q "WORKFLOW_TRIGGER fired" "${WF_LOG}" || die "workflow mock missing marker"
grep -q "ROUTING_CHANGE_CALLBACK fired" "${RT_LOG}" || die "routing mock missing marker"
grep -q "routing.updated" "${RT_LOG}" || die "routing payload missing event"
grep -q "DATA_DELETE_CALLBACK fired" "${DD_LOG}" || die "data-delete mock missing marker"
grep -q "data.deleted" "${DD_LOG}" || die "data-delete payload missing event"
ok "Phase A: all three echo mocks printed"

echo ""
echo "----- workflow.log -----"
cat "${WF_LOG}"
echo "----- routing.log -----"
cat "${RT_LOG}"
echo "----- data_delete.log -----"
cat "${DD_LOG}"
echo ""

# ---------- Phase B: live HTTP (optional) ----------
if [[ -n "${SMOKE_BASE_URL}" ]]; then
  info "Phase B: live APIs at ${SMOKE_BASE_URL}"
  command -v curl >/dev/null || die "curl required for Phase B"
  BASE="${SMOKE_BASE_URL%/}"

  # Clear logs so we can see new fires (workflow only via /flush)
  : >"${WF_LOG}"

  code="$(
    curl -sS -o "${LOG_DIR}/flush.json" -w '%{http_code}' \
      -X POST "${BASE}/flush" \
      -H "Authorization: Bearer ${PROXY_API_KEY}" \
      -H "Content-Type: application/json" \
      -d '{"trigger_workflow": true}' || true
  )"
  echo "POST /flush → HTTP ${code}"
  cat "${LOG_DIR}/flush.json" 2>/dev/null || true
  echo
  [[ "${code}" == "200" ]] || die "flush failed (is tracer using same WORKFLOW_TRIGGER_SCRIPT?)"

  # Note: live tracer process must have WORKFLOW_TRIGGER_SCRIPT pointed at our mock
  # for the workflow log to update; if tracer is a separate container, Phase B only
  # checks HTTP 200 unless you mount these mocks into the container.
  sleep 1
  if [[ -s "${WF_LOG}" ]] && grep -q "WORKFLOW_TRIGGER fired" "${WF_LOG}"; then
    ok "Phase B: flush triggered workflow mock"
  else
    echo "⚠ Phase B: /flush returned ${code}, but workflow mock log unchanged."
    echo "  (Expected if tracer runs elsewhere without these mock env mounts.)"
  fi

  if [[ -n "${ADMIN_API_KEY}" ]]; then
    code="$(
      curl -sS -o "${LOG_DIR}/admin.json" -w '%{http_code}' \
        -X GET "${BASE}/admin/routing" \
        -H "Authorization: Bearer ${ADMIN_API_KEY}" || true
    )"
    echo "GET /admin/routing → HTTP ${code}"
    head -c 400 "${LOG_DIR}/admin.json" 2>/dev/null || true
    echo
    [[ "${code}" == "200" ]] || die "admin routing GET failed"
    ok "Phase B: admin routing reachable"
  else
    echo "⚠ ADMIN_API_KEY unset; skip admin routing smoke"
  fi
else
  info "Phase B skipped (set SMOKE_BASE_URL to hit live /flush)"
fi

ok "smoke-callbacks finished"
if [[ "${SMOKE_KEEP_LOGS:-0}" != "1" ]]; then
  # keep dir path printed above; do not delete so user can inspect unless asked
  :
fi
echo "logs kept at: ${LOG_DIR}"
