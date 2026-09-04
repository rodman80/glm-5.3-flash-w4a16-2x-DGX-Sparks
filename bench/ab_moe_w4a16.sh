#!/usr/bin/env bash
# ab_moe_w4a16.sh — A/B marlin vs flashinfer_cutlass on W4A16-MTP.
# Same checkpoint, same image, only MOE_BACKEND changes (edited in .env,
# since start.sh/launch source .env and would clobber an export).
# Each cold boot ~10min. Alternating order to decorrelate thermal drift.
# Usage: nohup ./bench/ab_moe_w4a16.sh > /tmp/ab-moe.log 2>&1 &
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"
DATE="${DATE:-$(date +%Y%m%d-%H%M)}"
OUTBASE="$REPO_DIR/benchmarks/ab-moe-$DATE"
mkdir -p "$OUTBASE"
exec > >(tee -a "$OUTBASE/run.log") 2>&1
BASE="http://127.0.0.1:8000"

log(){ echo "[$(date '+%F %T')] $*"; }
boot_backend(){
  local backend="$1"
  log "boot MOE_BACKEND=$backend"
  sed -i "s|^MOE_BACKEND=.*|MOE_BACKEND=$backend|" .env
  grep "^MOE_BACKEND=" .env
  SKIP_PULL=1 SKIP_SYNC=1 ./start.sh restart 2>&1 | tail -n 3
}
wait_ready(){
  for _ in $(seq 1 180); do
    curl -sf --max-time 5 "$BASE/v1/models" >/dev/null 2>&1 && { log "engine ready"; return 0; }
    sleep 15
  done
  log "timeout"; return 1
}
run_label(){
  local label="$1"
  log "== $label =="
  ./bench/bench_config.sh "$label" 2>&1 | tail -n 12
  rm -rf "$OUTBASE/$label"
  cp -r "benchmarks/$label" "$OUTBASE/$label" 2>/dev/null || true
}

log "==== ab_moe_w4a16 start $DATE ===="
n=1
for backend in marlin flashinfer_cutlass marlin flashinfer_cutlass; do
  label="r${n}-${backend}"
  boot_backend "$backend" || { log "boot fail $label"; n=$((n+1)); continue; }
  wait_ready || { log "wait fail $label"; n=$((n+1)); continue; }
  run_label "$label"
  n=$((n+1))
done

# restore auto
sed -i "s|^MOE_BACKEND=.*|MOE_BACKEND=|" .env
log "==== done -> $OUTBASE (.env restored to auto) ===="
echo "--- soak medians ---"
grep -h decode_tok_s_median "$OUTBASE"/r*/soak.json 2>/dev/null || true
echo "--- aggregates ---"
for f in "$OUTBASE"/r*/c1.json "$OUTBASE"/r*/c2.json "$OUTBASE"/r*/c6.json; do
  echo "$f: $(grep -o '"aggregate_tok_s": [0-9.]*' "$f" 2>/dev/null)"
done
echo "--- acceptance ---"
grep -h "ratio=" "$OUTBASE"/r*/acceptance.txt 2>/dev/null || true
