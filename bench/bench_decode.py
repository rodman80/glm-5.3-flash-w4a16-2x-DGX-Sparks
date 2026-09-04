#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.request

p = argparse.ArgumentParser(description="Small streaming decode benchmark for the GLM-5.3 A/B test")
p.add_argument("--base", default="http://127.0.0.1:8000/v1")
p.add_argument("--model", default="glm-5.3-flash")
p.add_argument("--runs", type=int, default=5)
p.add_argument("--max-tokens", type=int, default=512)
p.add_argument("--prompt", default="Write a compact Python function that parses a log line, validates fields, and returns a typed dictionary. Explain edge cases after the code.")
a = p.parse_args()

url = a.base.rstrip("/") + "/chat/completions"
payload = {
    "model": a.model,
    "messages": [{"role": "user", "content": a.prompt}],
    "temperature": 0,
    "max_tokens": a.max_tokens,
    "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"enable_thinking": False},
}
raw = json.dumps(payload).encode()
results = []

for n in range(1, a.runs + 1):
    req = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    first = None
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for b in r:
            line = b.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            evt = json.loads(line[6:])
            choices = evt.get("choices") or []
            if choices and first is None:
                d = choices[0].get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    first = time.perf_counter()
            usage = evt.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
    t1 = time.perf_counter()
    ttft = (first - t0) if first else float("nan")
    decode_s = max(t1 - (first or t0), 1e-9)
    decode_tokens = max((completion_tokens or 0) - 1, 0)
    tok_s = decode_tokens / decode_s
    row = {"run": n, "ttft_s": round(ttft, 4), "total_s": round(t1-t0, 4), "completion_tokens": completion_tokens, "decode_tok_s": round(tok_s, 3)}
    results.append(row)
    print(json.dumps(row), flush=True)

vals = [x["decode_tok_s"] for x in results]
ttfts = [x["ttft_s"] for x in results]
print(json.dumps({
    "summary": True,
    "runs": len(results),
    "decode_tok_s_median": round(statistics.median(vals), 3),
    "decode_tok_s_mean": round(statistics.mean(vals), 3),
    "ttft_s_median": round(statistics.median(ttfts), 4),
}), flush=True)
