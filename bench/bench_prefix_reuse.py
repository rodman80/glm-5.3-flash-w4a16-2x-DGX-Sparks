#!/usr/bin/env python3
"""bench_prefix_reuse.py — measures prefix-cache over agent-style turns.

Sends the SAME long prompt 2x (like consecutive turns resending
system+context) and measures:
  - TTFT req1 (cold) vs req2 (expects a prefix hit)
  - /metrics deltas: prefix_cache_hits_total / queries_total
  - decode tok/s on both

The cache is block-granular (BLOCK_SIZE=2304): only identical full blocks hit.
~5000-token prompt => 2 full blocks (4608) + remainder.

Usage: python3 bench_prefix_reuse.py --base http://127.0.0.1:8000/v1 [--prompt-tokens 5000]
"""
import argparse
import json
import re
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://127.0.0.1:8000/v1")
p.add_argument("--model", default="glm-5.3-flash")
p.add_argument("--prompt-tokens", type=int, default=5000)
p.add_argument("--max-tokens", type=int, default=128)
a = p.parse_args()

# ~4 chars/token: 5000 tok ~= 20KB. Stable prefix up front (system+repo),
# identical on every request — like an agent resending context.
SYSTEM = (
    "You are an expert coding agent. Follow the repository conventions strictly. "
    "Output only code changes with brief explanations. Respect the existing "
    "architecture, naming, and error handling patterns described below.\n"
)
UNIT = (
    "File src/module.py defines class Handler with methods for parsing, "
    "validation, and serialization. Invariants: inputs are sanitized at the "
    "boundary; errors propagate as typed exceptions; logging uses structlog.\n"
)
n_units = max(1, (a.prompt_tokens * 4 - len(SYSTEM)) // len(UNIT))
PROMPT = SYSTEM + UNIT * n_units
print(f"prompt chars={len(PROMPT)} (~{len(PROMPT)//4} tok target={a.prompt_tokens})")


def metrics():
    out = {}
    with urllib.request.urlopen(a.base.rstrip("/v1") + "/metrics", timeout=30) as r:
        text = r.read().decode()
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"^(vllm:prefix_cache_\w+)(?:\{[^}]*\})? ([0-9.eE+]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def chat(prompt, temp=0):
    body = json.dumps({
        "model": a.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": a.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        a.base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    first = None
    ct = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for b in r:
            line = b.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            evt = json.loads(line[6:])
            if (evt.get("choices") or [{}])[0].get("delta", {}).get("content") and first is None:
                first = time.perf_counter()
            u = evt.get("usage") or {}
            if u.get("completion_tokens") is not None:
                ct = int(u["completion_tokens"])
    t1 = time.perf_counter()
    return {"ttft_s": round((first or t1) - t0, 3), "total_s": round(t1 - t0, 2),
            "tokens": ct, "decode_tok_s": round(max(ct - 1, 0) / max(t1 - (first or t0), 1e-9), 2)}


for i in (1, 2):
    m0 = metrics()
    r = chat(PROMPT)
    m1 = metrics()
    dh = m1.get("vllm:prefix_cache_hits_total", 0) - m0.get("vllm:prefix_cache_hits_total", 0)
    dq = m1.get("vllm:prefix_cache_queries_total", 0) - m0.get("vllm:prefix_cache_queries_total", 0)
    print(json.dumps({"req": i, **r, "prefix_hits": dh, "prefix_queries": dq,
                       "hit_rate": round(dh / dq, 3) if dq else None}))
