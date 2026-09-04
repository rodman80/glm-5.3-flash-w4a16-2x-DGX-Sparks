#!/usr/bin/env python3
"""bench_c.py — concurrent decode-throughput benchmark (C1..Cn).

Fires `--conc` simultaneous streaming chat completions (512 tokens each,
coding prompt with per-request variation to avoid prefix-cache sharing)
and reports aggregate throughput: total completion tokens / wall time from
first request start to last request finish.

Usage: python3 bench_c.py --conc 6 [--runs 6] [--max-tokens 512]
"""
import argparse
import json
import threading
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://127.0.0.1:8000/v1")
p.add_argument("--model", default="glm-5.3-flash")
p.add_argument("--conc", type=int, default=6)
p.add_argument("--runs", type=int, default=0, help="total requests (default: conc)")
p.add_argument("--max-tokens", type=int, default=512)
a = p.parse_args()
runs = a.runs if a.runs > 0 else a.conc

PROMPTS = [
    "Write a compact Python function that parses a log line, validates fields, and returns a typed dictionary. Explain edge cases after the code.",
    "Write a compact Python function that normalizes a CSV row, coerces types, and returns a dataclass. Explain edge cases after the code.",
    "Write a compact Python function that parses an INI-style config, validates keys, and returns a dict. Explain edge cases after the code.",
    "Write a compact Python function that parses a JSON payload, validates required fields, and returns a typed object. Explain edge cases after the code.",
    "Write a compact Python function that parses a key=value env string, validates values, and returns a dict. Explain edge cases after the code.",
    "Write a compact Python function that parses a syslog line, extracts the fields, and returns a struct. Explain edge cases after the code.",
    "Write a compact Python function that parses a semicolon-separated record, validates fields, and returns a tuple. Explain edge cases after the code.",
    "Write a compact Python function that parses a tab-separated line, validates columns, and returns a list. Explain edge cases after the code.",
]


def worker(i, out, lock):
    prompt = PROMPTS[i % len(PROMPTS)]
    payload = {
        "model": a.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": a.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        a.base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    completion_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for b in r:
                line = b.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                evt = json.loads(line[6:])
                usage = evt.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
    except Exception as e:  # noqa: BLE001
        completion_tokens = 0
    t1 = time.perf_counter()
    with lock:
        out.append({"req": i, "tokens": completion_tokens, "wall_s": round(t1 - t0, 2)})


out = []
lock = threading.Lock()
t_start = time.perf_counter()
threads = [threading.Thread(target=worker, args=(i, out, lock)) for i in range(runs)]
for t in threads:
    t.start()
for t in threads:
    t.join()
t_end = time.perf_counter()

total_tokens = sum(r["tokens"] for r in out)
wall = t_end - t_start
agg = total_tokens / wall if wall > 0 else float("nan")
print(json.dumps(
    {"conc": a.conc, "requests": runs, "total_tokens": total_tokens,
     "wall_s": round(wall, 2), "aggregate_tok_s": round(agg, 2),
     "per_request": out}, indent=1))
