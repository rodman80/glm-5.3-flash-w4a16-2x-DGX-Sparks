#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";cd "$(dirname "$D")";NAME="${NAME:-vllm_glm53_w4a16}";OUT="${1:-benchmarks/kb1-$(date +%Y%m%d-%H%M%S)}";mkdir -p "$OUT"
P=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py
if ! docker exec "$NAME" sh -lc 'test "${GLM53_KB_MOE_PROFILE:-0}" = 1 && grep -q "glm53-kb-moe" '"$P";then echo "KB1 overlay not active; set GLM53_KB_MOE_PROFILE=1 and restart" >&2;exit 2;fi
run(){ local l="$1";shift;local s="$(date -Iseconds)";"$@"|tee "$OUT/$l.bench.txt";sleep 2;docker logs --since "$s" "$NAME">"$OUT/$l.log" 2>&1||true;python3 bench/kb1_parse_moe_profile.py "$OUT/$l.log" --label "$l"|tee "$OUT/$l.summary.txt";}
run c1 python3 bench/bench_decode.py --runs 3 --max-tokens 1024
run c6 python3 bench/bench_c.py --conc 6 --runs 6 --max-tokens 512
echo "KB1 results: $OUT"
