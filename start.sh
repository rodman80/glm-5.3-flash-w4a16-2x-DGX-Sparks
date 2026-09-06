#!/usr/bin/env bash
# start.sh — 2x DGX Spark orchestrator for canada-quant/glm-5.3-w4a16-mtp
# 1) preflight 2) pull image on both nodes 3) HF download if missing 4) rsync worker 5) launch TP=2 6) health poll
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ENV_FILE="${ENV_FILE:-.env}"  # ENV_FILE=.env.base for baseline A/B boots
export ENV_FILE
[ -f "$SCRIPT_DIR/$ENV_FILE" ] || { cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/$ENV_FILE"; echo "[start] $ENV_FILE created from .env.example — edit HEAD_IP/WORKER_IP if needed"; }
set -a; source "$SCRIPT_DIR/$ENV_FILE"; set +a

MODEL_HOST_PATH="${MODEL_HOST_PATH:-/var/tmp/glm-5.3-flash-w4a16-mtp}"
DFLASH_HOST_PATH="${DFLASH_HOST_PATH:-/var/tmp/models/GLM-5.3-Flash-DFlash2}"
WORKER_SSH="${WORKER_SSH:-${WORKER_USER:+${WORKER_USER}@}${WORKER_IP:-10.100.24.1}}"
# default without user -> let ssh config resolve
if [ -z "$WORKER_SSH" ]; then WORKER_SSH="${WORKER_IP:-10.100.24.1}"; fi
HEAD_IP="${HEAD_IP:-10.100.24.2}"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"

check_swappiness() {
  local node="$1" val
  if [ "$node" = "head" ]; then val=$(cat /proc/sys/vm/swappiness 2>/dev/null || echo "?")
  else val=$(ssh -o ConnectTimeout=10 "$WORKER_SSH" "cat /proc/sys/vm/swappiness 2>/dev/null || echo ?" 2>/dev/null || echo "?"); fi
  if [ "$val" != "?" ] && [ "$val" -gt 10 ] 2>/dev/null; then
    echo "[warn] $node swappiness=$val (recommended 0-10 for DGX Spark; UVM livelock risk under load)" >&2
    if [ "$node" = "head" ]; then sudo sysctl vm.swappiness=10 2>/dev/null && echo "[fix] head swappiness -> 10" || echo "[fix] run: sudo sysctl vm.swappiness=10 ($node)" >&2
    else ssh -o ConnectTimeout=10 "$WORKER_SSH" "sudo sysctl vm.swappiness=10 2>/dev/null && echo '[fix] worker swappiness -> 10' || echo '[fix] run: sudo sysctl vm.swappiness=10 (worker)'" 2>&1 | sed 's/^/[worker] /'; fi
  else echo "[ok] $node swappiness=$val"; fi
}

check_disk() {
  local need_gb=190
  local avail
  avail=$(df -BG /var/tmp 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
  if [ -n "$avail" ] && [ "$avail" -lt "$need_gb" ] 2>/dev/null; then
    echo "[warn] /var/tmp has ${avail}G free, needs ~${need_gb}G for W4A16-MTP (178G+2.3G)" >&2
  else echo "[ok] head disk: ${avail:-?}G free"; fi
  local w_avail
  w_avail=$(ssh -o ConnectTimeout=10 "$WORKER_SSH" "df -BG /var/tmp 2>/dev/null | awk 'NR==2{print \$4}' | tr -d 'G'" 2>/dev/null || echo "?")
  if [ "$w_avail" != "?" ] && [ "$w_avail" -lt "$need_gb" ] 2>/dev/null; then
    echo "[warn] worker /var/tmp has ${w_avail}G free" >&2
  else echo "[ok] worker disk: ${w_avail}G free"; fi
}

stop_conflicting() {
  echo "[stop] removing conflicting vllm containers on both nodes"
  # head
  for n in $(docker ps -a --format '{{.Names}}' 2>/dev/null | grep -E 'vllm_glm53|vllm_' || true); do
    echo "  head: docker rm -f $n"; docker rm -f "$n" 2>/dev/null || true
  done
  # worker
  ssh -o ConnectTimeout=10 "$WORKER_SSH" '
    for n in $(docker ps -a --format "{{.Names}}" 2>/dev/null | grep -E "vllm_glm53|vllm_" || true); do
      echo "  worker: docker rm -f $n"; docker rm -f "$n" 2>/dev/null || true
    done
  ' 2>&1 | sed 's/^/[worker] /' || true
}

cmd="${1:-start}"
case "$cmd" in
  start|restart)
    if [ "$cmd" = "restart" ]; then
      stop_conflicting
      sleep 2
    fi
    # preflight
    echo "[preflight] docker + ssh + disk + swappiness"
    docker info >/dev/null || { echo "docker not running on head" >&2; exit 1; }
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "docker info >/dev/null && echo worker:docker:ok" || { echo "ssh/worker docker failed ($WORKER_SSH)" >&2; exit 1; }
    check_disk
    check_swappiness head
    check_swappiness worker

    # pull image
    if [ "${SKIP_PULL:-0}" != "1" ]; then
      echo "[pull] head: $IMAGE"
      docker pull "${IMAGE}" 2>&1 | tail -n 5 || docker pull "${IMAGE_FALLBACK:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}" 2>&1 | tail -n 5 || true
      echo "[pull] worker"
      if ! ssh -o ConnectTimeout=30 "$WORKER_SSH" "docker pull ${IMAGE} 2>&1 | tail -n 5 || docker pull ${IMAGE_FALLBACK:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2} 2>&1 | tail -n 5" 2>&1 | tail -n 20; then
        echo "[ship] worker has no GHCR — copying via save/load"
        docker save --platform linux/arm64 "$IMAGE" 2>/dev/null | ssh "$WORKER_SSH" "docker load" 2>&1 | tail -n 10 || \
        docker save "$IMAGE" | ssh "$WORKER_SSH" "docker load" 2>&1 | tail -n 10
      fi
    else echo "[pull] SKIP_PULL=1 — using local images"; fi

    # download if missing
    if [ ! -f "$MODEL_HOST_PATH/config.json" ]; then
      echo "[download] weights not found — downloading ~178 GiB"
      "$SCRIPT_DIR/download.sh"
    else
      echo "[download] weights already at $MODEL_HOST_PATH ($(find "$MODEL_HOST_PATH" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l) shards)"
    fi

    # sync to worker (model + drafter + patch + repo)
    if [ "${SKIP_SYNC:-0}" != "1" ]; then
      echo "[sync] rsync model/drafter/patch/repo to worker $WORKER_SSH"
      ssh -o ConnectTimeout=10 "$WORKER_SSH" "mkdir -p \"\$HOME/patches\" \"$MODEL_HOST_PATH\" \"$DFLASH_HOST_PATH\"" 2>&1 | sed 's/^/[worker] /' || true
      # top-k patch
      if [ -n "${PATCH_KPOOL_HOST:-}" ] && [ -f "$PATCH_KPOOL_HOST" ]; then
        rsync -av --progress "$PATCH_KPOOL_HOST" "$WORKER_SSH:$PATCH_KPOOL_HOST" 2>&1 | tail -n 5 || \
        rsync -av --progress "$PATCH_KPOOL_HOST" "$WORKER_SSH:~/patches/sparse_attn_indexer_kpool.py" 2>&1 | tail -n 5 || true
      fi
      # model and drafter (can take ~10 min for 180G over RoCE on first sync)
      echo "[sync] model (180G) — may take minutes on first sync"
      rsync -av --progress "$MODEL_HOST_PATH/" "$WORKER_SSH:$MODEL_HOST_PATH/" 2>&1 | tail -n 20 || echo "WARN: model rsync failed — copy manually" >&2
      if [ -d "$DFLASH_HOST_PATH" ]; then
        rsync -av --progress "$DFLASH_HOST_PATH/" "$WORKER_SSH:$DFLASH_HOST_PATH/" 2>&1 | tail -n 10 || true
      fi
      if [ -f "$MODEL_HOST_PATH/chat_template_mm.jinja" ]; then
        rsync -av "$MODEL_HOST_PATH/chat_template_mm.jinja" "$WORKER_SSH:$MODEL_HOST_PATH/" 2>&1 | tail -n 3 || true
      fi
      # repo scripts (so the worker has the same launch.sh)
      rsync -av --progress --exclude '.git' --exclude '__pycache__' "$SCRIPT_DIR/" "$WORKER_SSH:$SCRIPT_DIR/" 2>&1 | tail -n 20 || \
        echo "[warn] repo rsync failed — worker will run via direct docker" >&2
    else echo "[sync] SKIP_SYNC=1 — skipping rsync"; fi

    # warn about still-running conflicting containers
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qE 'vllm_glm53'; then
      echo "[warn] a vllm_glm53 is still running — will fight for GPU/RAM. Stop with ./start.sh stop or run restart" >&2
      if [ "$cmd" != "restart" ]; then echo "[warn] continuing but may OOM (121G UMA) — use restart to clean" >&2; fi
    fi
    if [ "$cmd" = "start" ]; then stop_conflicting; sleep 2; fi

    # GB10 ritual
    echo "[ritual] drop_caches on both nodes"
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 && echo "[ok] head drop_caches" || echo "[warn] sudo drop_caches failed on head (try manually)" >&2
    ssh -o ConnectTimeout=10 "$WORKER_SSH" "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 && echo '[ok] worker drop_caches' || echo '[warn] sudo drop_caches failed on worker'" 2>&1 | sed 's/^/[worker] /' || true

    echo "[launch] worker rank=1"
    if ssh -o ConnectTimeout=30 "$WORKER_SSH" "bash -lc 'cd \"$SCRIPT_DIR\" && ENV_FILE=\"$ENV_FILE\" ./launch-glm53-w4a16-tp2-dflash2.sh 1'" 2>&1 | tail -n 30; then
      echo "[launch] worker ok"
    else
      echo "[launch] worker via script failed — trying direct docker (fallback)"
      # inline fallback over ssh: runs the same docker run without needing the script on the worker
      ssh -T "$WORKER_SSH" "bash -lc 'cd \"$SCRIPT_DIR\" 2>/dev/null && ./launch-glm53-w4a16-tp2-dflash2.sh 1 || echo fallback:worker:script-not-found'" 2>&1 | tail -n 30 || true
      # check the container came up
      sleep 5
      if ! ssh -o ConnectTimeout=10 "$WORKER_SSH" "docker ps --format '{{.Names}}' | grep -q vllm_glm53_w4a16" 2>&1; then
        echo "[error] worker did not come up — check ssh and $SCRIPT_DIR on the worker" >&2
        exit 1
      fi
    fi
    echo "[launch] waiting 25s for NCCL rendezvous"
    sleep 25
    echo "[launch] head rank=0"
    "$SCRIPT_DIR/launch-glm53-w4a16-tp2-dflash2.sh" 0

    echo "[health] polling http://$HEAD_IP:$PORT/health up to ${READY_TIMEOUT}s (cold 6-8 min, 180G shards)"
    t=0
    until curl -sf "http://$HEAD_IP:$PORT/health" >/dev/null 2>&1; do
      sleep 20; t=$((t+20))
      echo "  ... ${t}s / ${READY_TIMEOUT}s (still loading shards)"
      docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep -q vllm_glm53_w4a16 || { echo "head container died"; docker logs vllm_glm53_w4a16 2>&1 | tail -n 120; exit 1; }
      if [ "$t" -ge "$READY_TIMEOUT" ]; then echo "timeout on /health — head docker logs:" >&2; docker logs vllm_glm53_w4a16 2>&1 | tail -n 120 >&2; exit 1; fi
    done
    echo "[health] OK — serving at http://$HEAD_IP:$PORT/v1  (model: $SERVED_MODEL_NAME)"
    if [ "${LOCK_CLOCKS:-1}" = "1" ]; then
      echo "[clocks] locking GB10 clocks to max (perf recipe step)"
      "$SCRIPT_DIR/clocks.sh" "${CLOCK_MHZ:-2400}" 2>&1 | tail -n 6 || echo "[warn] clocks.sh failed — serving continues at stock clocks" >&2
    else echo "[clocks] LOCK_CLOCKS=0 — skipped (stock clocks)"; fi
    echo "Test: curl http://$HEAD_IP:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"2+2=?\"}],\"max_tokens\":40,\"chat_template_kwargs\":{\"enable_thinking\":false}}'"
    ;;
  stop)
    stop_conflicting
    ;;
  status)
    echo "== head =="; docker ps --format '{{.Names}} {{.Status}} {{.Ports}}' 2>/dev/null | grep -E 'vllm|glm' || echo "(no vllm on head)"
    curl -sf "http://$HEAD_IP:$PORT/health" 2>&1 | head -n 5 && echo "health: OK" || echo "health: NOK (port $PORT)"
    echo "== worker ($WORKER_SSH) =="; ssh -o ConnectTimeout=10 "$WORKER_SSH" "docker ps --format '{{.Names}} {{.Status}} {{.Ports}}' 2>/dev/null | grep -E 'vllm|glm' || echo '(no vllm on worker)'; curl -sf http://$HEAD_IP:$PORT/health 2>&1 | head -n 3 && echo health:OK || echo health:NOK" 2>&1 | sed 's/^/[worker] /' || true
    echo "== local images =="; docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>/dev/null | grep -i glm | head || true
    ;;
  logs)
    if [ "${2:-}" = "worker" ]; then ssh -T "$WORKER_SSH" "docker logs -f vllm_glm53_w4a16" 2>&1 | tail -n 200; else docker logs -f vllm_glm53_w4a16 2>&1 | tail -n 200; fi
    ;;
  download)
    exec "$SCRIPT_DIR/download.sh"
    ;;
  validate)
    exec "$SCRIPT_DIR/validate.sh"
    ;;
  *)
    echo "usage: $0 {start|restart|stop|status|logs [worker]|download|validate}" >&2; exit 2;;
esac
