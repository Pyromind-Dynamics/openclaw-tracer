#!/bin/bash
# Mock ROUTING_CHANGE_CALLBACK_SCRIPT — argv[1] = JSON tempfile.
echo "[smoke-mock] ROUTING_CHANGE_CALLBACK fired at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ -n "${1:-}" && -f "$1" ]]; then
  echo "[smoke-mock] payload:"
  cat "$1"
  echo
else
  echo "[smoke-mock] (no payload file)"
fi
