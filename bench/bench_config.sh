#!/usr/bin/env bash
# bench_config.sh — before/after baseline for W4A16-MTP 2x Spark.
# Ported from glm53-redhat-nvfp4-dgx-spark/bench_config.sh:
#   NAME=vllm_glm53_w4a16, port 8000, no hardcoded worker dependency.
# Usage: ./bench/bench_config.sh <label>  (e.g. baseline-8000-marlin)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

LABEL="${1:?usage: bench_config.sh <label>}"
NAME="${NAME:-vllm_glm53_w4a16}"
BASE="${BASE:-http://127.0.0.1:8000}"
OUTDIR="$REPO_DIR/benchmarks/$LABEL"
mkdir -p "$OUTDIR"
BENCHLOG="$OUTDIR/bench.log"
exec > >(tee -a "$BENCHLOG") 2>&1

fail() { echo "[$(date '+%T')] FAIL: $*"; }
echo "=== bench_config $LABEL $(date '+%F %T') base=$BASE ==="

# 1. ready?
ready=0
for _ in $(seq 1 30); do
  if curl -sf --max-time 5 "$BASE/v1/models" >/dev/null 2>&1; then ready=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    fail "container $NAME not running"; docker logs --tail 20 "$NAME" 2>&1 | tail -20 || true; exit 1
  fi
  sleep 10
done
(( ready )) || { fail "engine not ready"; exit 1; }
echo "engine ready"

# 2. gate + signatures
./check-dflash2.sh > "$OUTDIR/gate.txt" 2>&1 || true
cat "$OUTDIR/gate.txt"
./verify-running-backend.sh > "$OUTDIR/backend.txt" 2>&1 || true
cat "$OUTDIR/backend.txt"
docker logs "$NAME" > "$OUTDIR/engine.log" 2>&1 || true
EL="$OUTDIR/engine.log"
kv_tokens=$(grep -oE "GPU KV cache size: [0-9,]+ tokens" "$EL" | tail -1 | grep -oE "[0-9,]+" | tr -d ',')
moe_line=$(grep -E "WNA16 MoE backend|MarlinExperts" "$EL" | tail -1 || true)
echo "signatures: kv_tokens=${kv_tokens:-n/a} moe=[${moe_line:-n/a}]"
echo "memfree kB head=$(awk '/^MemFree:/{print $2}' /proc/meminfo)"

# 3. warmup (discarded — 1st request after boot doesn't count)
echo "warmup (not counted)..."
python3 bench_decode.py --base "$BASE/v1" --runs 1 > "$OUTDIR/warmup.json" 2>&1 || fail "warmup failed"

# 4. soak 3x C1
echo "soak x3 C1..."
python3 bench_decode.py --base "$BASE/v1" --runs 3 > "$OUTDIR/soak.json" 2>&1 || fail "soak failed"
cat "$OUTDIR/soak.json"

# 5. C1/C2/C6
for c in 1 2 6; do
  echo "bench_c C$c..."
  python3 bench_c.py --base "$BASE/v1" --conc "$c" --runs "$c" --max-tokens 512 > "$OUTDIR/c$c.json" 2>&1 || fail "c$c failed"
  tail -n 5 "$OUTDIR/c$c.json"
done

# 6. acceptance
python3 acceptance_ratio.py "$BASE" > "$OUTDIR/acceptance.txt" 2>&1 || fail "acceptance missing"
cat "$OUTDIR/acceptance.txt" | tail -n 8

# 7. P1 determinism
p1=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"One sentence: 17*23 and the capital of Japan?"}],"max_tokens":120,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message'].get('content') or '')" 2>/dev/null)
echo "P1: $p1"
echo "$p1" > "$OUTDIR/p1.txt"

echo "=== bench_config $LABEL done -> $OUTDIR ==="
