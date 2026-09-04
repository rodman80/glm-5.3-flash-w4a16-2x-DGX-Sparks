#!/usr/bin/env bash
set -euo pipefail
NAME="${NAME:-vllm_glm53_w4a16}"
LOG="$(docker logs "$NAME" 2>&1 || true)"
echo "== MoE lines =="
printf '%s\n' "$LOG" | grep -iE "WNA16 MoE backend|MarlinExperts|flashinfer_cutlass|compressedtensors" | tail -n 10 || true
echo
if printf '%s\n' "$LOG" | grep -qF "'MARLIN' WNA16 MoE backend"; then
  echo "BACKEND=marlin (pinned in .env)"
elif printf '%s\n' "$LOG" | grep -qF "flashinfer_cutlass"; then
  echo "BACKEND=flashinfer_cutlass"
else
  echo "BACKEND=unknown" >&2; exit 1
fi
