#!/usr/bin/env bash
# verify_final.sh — per-boot measurement suite for the 3v3 final verify.
# Usage: ./bench/verify_final.sh <outdir>
# Runs: P1 gate + acceptance + 9x C1-512 + 3x C6-512 + 2x 100k-prefill.
# Appends JSONL rows + summary to <outdir>/.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
OUT_ARG="${1:?usage: verify_final.sh <outdir>}"
BASE="${BASE:-http://127.0.0.1:8000}"
case "$OUT_ARG" in /*) OUT="$OUT_ARG";; *) OUT="$(pwd)/$OUT_ARG";; esac
mkdir -p "$OUT"
cd "$SCRIPT_DIR"

echo "=== verify $(date '+%F %T') base=$BASE -> $OUT ===" | tee "$OUT/run.log"

# P1 gate
p1=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"One sentence: 17*23 and the capital of Japan?"}],"max_tokens":120,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message'].get('content') or '')" 2>/dev/null)
echo "P1: $p1" | tee -a "$OUT/run.log"
echo "$p1" > "$OUT/p1.txt"
echo "$p1" | grep -q "391" || { echo "P1 GATE FAILED" | tee -a "$OUT/run.log"; exit 1; }
echo "$p1" | grep -qi "tokyo" || { echo "P1 GATE FAILED" | tee -a "$OUT/run.log"; exit 1; }
echo "P1 gate: PASS" | tee -a "$OUT/run.log"

# warmup (discarded)
python3 bench_decode.py --base "$BASE/v1" --runs 1 --max-tokens 128 > /dev/null 2>&1 || true

# acceptance (cumulative since boot — record only)
python3 acceptance_ratio.py "$BASE" > "$OUT/acceptance.txt" 2>&1 || true
tail -n 3 "$OUT/acceptance.txt" | tee -a "$OUT/run.log"

# 9x C1-512
python3 bench_decode.py --base "$BASE/v1" --runs 9 --max-tokens 512 > "$OUT/c1x9.json" 2>&1 || { echo "C1 FAILED" | tee -a "$OUT/run.log"; exit 1; }
grep -E '"run"|median' "$OUT/c1x9.json" | tee -a "$OUT/run.log"

# 3x C6-512
for i in 1 2 3; do
  python3 bench_c.py --base "$BASE/v1" --conc 6 --runs 6 --max-tokens 512 > "$OUT/c6_$i.json" 2>&1 || { echo "C6_$i FAILED" | tee -a "$OUT/run.log"; exit 1; }
  grep "aggregate_tok_s" "$OUT/c6_$i.json" | tee -a "$OUT/run.log"
  sleep 5
done

# 2x 100k cold prefill (unique salt each)
python3 - "$OUT/prefill100k.json" <<'EOF' 2>&1 | tee -a "$OUT/run.log"
import json, sys, time, urllib.request, uuid
out = sys.argv[1]
BASE = "http://127.0.0.1:8000"
def tok_count(messages):
    body = {"model": "glm-5.3-flash", "messages": messages}
    req = urllib.request.Request(BASE + "/tokenize", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return int(json.loads(urllib.request.urlopen(req, timeout=180).read().decode())["count"])
rows = []
for i in range(2):
    salt = f"VERIFY100k-{uuid.uuid4()}"
    n = 100000
    text = salt + "\n" + "the " * n + "\nReply with OK."
    got = tok_count([{"role": "user", "content": text}])
    n2 = max(n + (100000 - got), 1)
    text = salt + "\n" + "the " * n2 + "\nReply with OK."
    got = tok_count([{"role": "user", "content": text}])
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": text}],
            "temperature": 0, "max_tokens": 8, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter(); first = None; pt = 0
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=900) as r:
        for b in r:
            line = b.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            evt = json.loads(line[6:])
            ch = evt.get("choices") or []
            if ch and first is None:
                d = ch[0].get("delta") or {}
                if d.get("content"):
                    first = time.perf_counter()
            u = evt.get("usage") or {}
            if u.get("prompt_tokens"):
                pt = int(u["prompt_tokens"])
    row = {"run": i, "prompt_tokens": pt, "ttft_s": round(first - t0, 2),
           "prefill_tok_s": round(pt / (first - t0), 1)}
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(rows, open(out, "w"), indent=1)
EOF
echo "=== verify done -> $OUT ===" | tee -a "$OUT/run.log"
