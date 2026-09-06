#!/usr/bin/env bash
# clocks.sh — trava os clocks da GPU GB10 para serving (etapa da receita).
# Sem sudo: usa container privilegiado (grupo docker) em cada no.
# Default GB10 fica ~2281MHz; travar em 2400 da ganho sem calor excessivo
# (medido: +5-7% decode, +2-3% prefill vs stock). Persiste entre restarts
# de container; reaplicar apos reset da GPU/reboot do host. Idempotente.
# Usage: ./clocks.sh [max_mhz|reset]   (default 2400; reset = stock auto-boost)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ENV_FILE="${ENV_FILE:-.env}"
[ -f "$SCRIPT_DIR/$ENV_FILE" ] && { set -a; source "$SCRIPT_DIR/$ENV_FILE"; set +a; }
IMAGE="${IMAGE:-radixark/vllm-glm53-flash:sm121-v11-dflash2}"
WORKER_SSH="${WORKER_SSH:-${WORKER_USER:+${WORKER_USER}@}${WORKER_IP:-10.100.24.1}}"
[ -z "$WORKER_SSH" ] && WORKER_SSH="${WORKER_IP:-10.100.24.1}"
MAXMHZ="${1:-2400}"
if [ "$MAXMHZ" = "reset" ]; then
  echo "[clocks] resetting to stock auto-boost (head + worker)"
  docker run --rm --privileged --gpus all --entrypoint '' "$IMAGE" nvidia-smi -rgc 2>&1 | tail -n 1 | sed 's/^/[head] /'
  ssh -o ConnectTimeout=30 "$WORKER_SSH" "docker run --rm --privileged --gpus all --entrypoint '' $IMAGE nvidia-smi -rgc 2>&1 | tail -n 1" 2>&1 | sed 's/^/[worker] /'
  sleep 4
  echo "[clocks] head:  $(nvidia-smi -q -d CLOCK 2>/dev/null | awk '/Graphics/{print $3; exit}')"
  exit 0
fi
sleep 4
echo "[clocks] locking to ${MAXMHZ}MHz (head + worker)"
docker run --rm --privileged --gpus all --entrypoint '' "$IMAGE" nvidia-smi -lgc "$MAXMHZ,$MAXMHZ" 2>&1 | tail -n 2 | sed 's/^/[head] /'
ssh -o ConnectTimeout=30 "$WORKER_SSH" "docker run --rm --privileged --gpus all --entrypoint '' $IMAGE nvidia-smi -lgc $MAXMHZ,$MAXMHZ 2>&1 | tail -n 2" 2>&1 | sed 's/^/[worker] /'
sleep 4
echo "[clocks] head:  $(nvidia-smi -q -d CLOCK 2>/dev/null | awk '/Graphics/{print $3; exit}')"
echo "[clocks] worker: $(ssh -o ConnectTimeout=15 "$WORKER_SSH" 'nvidia-smi -q -d CLOCK 2>/dev/null | grep -oE "[0-9]+ MHz" | head -n 1' 2>/dev/null)"
