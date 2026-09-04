#!/usr/bin/env python3
"""Reproduce decode-vs-prefill interference with two overlapping requests.

Request A starts a long streaming decode. As soon as A emits its first content,
request B starts a large cold prompt with max_tokens=1. If B's prompt processing
lasts for most/all of A's decode, A's average decode tok/s is a useful direct
measure of mixed-prefill interference.

Run the exact same command after booting policy=off, skip, 4, 8, 16.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://127.0.0.1:8000/v1")
p.add_argument("--model", default="glm-5.3-flash")
p.add_argument("--decode-tokens", type=int, default=1024)
p.add_argument(
    "--prefill-repeats",
    type=int,
    default=7000,
    help="repeat count for the cold prefill payload; raise until B overlaps all of A",
)
p.add_argument("--timeout", type=int, default=900)
a = p.parse_args()

url = a.base.rstrip("/") + "/chat/completions"
trigger = threading.Event()
results: dict[str, object] = {}


def stream_request(messages, max_tokens, on_first=None):
    payload = {
        "model": a.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    first = None
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
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
                    if on_first is not None:
                        on_first()
            usage = evt.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
    t1 = time.perf_counter()
    return {
        "t0": t0,
        "first": first,
        "t1": t1,
        "completion_tokens": completion_tokens,
    }


def run_prefill():
    trigger.wait()
    # Deliberately cold and long. Put the unique salt at the *front* so the
    # first cache block differs and automatic prefix caching cannot reuse the
    # giant repeated tail from a previous benchmark invocation.
    salt = f"run-{time.time_ns()} "
    unit = "Analyze this numbered observation carefully and preserve its order. "
    prompt = salt + (unit * a.prefill_repeats)
    b = stream_request([{"role": "user", "content": prompt}], 1)
    results["prefill"] = b


prefill_thread = threading.Thread(target=run_prefill, daemon=True)
prefill_thread.start()

a_prompt = (
    "Write a detailed Python implementation of an in-memory transactional key-value "
    "store with MVCC, snapshots, conflict detection, tests, and explanation. Continue "
    "until the requested token budget is exhausted."
)
a_result = stream_request(
    [{"role": "user", "content": a_prompt}],
    a.decode_tokens,
    on_first=trigger.set,
)
results["decode"] = a_result
prefill_thread.join(timeout=a.timeout)

A = results["decode"]
assert isinstance(A, dict)
a_first = A.get("first")
a_t1 = float(A["t1"])
a_t0 = float(A["t0"])
a_tokens = int(A.get("completion_tokens") or 0)
decode_s = max(a_t1 - float(a_first or a_t0), 1e-9)
decode_tps = max(a_tokens - 1, 0) / decode_s

out = {
    "decode_completion_tokens": a_tokens,
    "decode_ttft_s": round((float(a_first) - a_t0) if a_first else float("nan"), 3),
    "decode_s_after_first": round(decode_s, 3),
    "decode_tok_s": round(decode_tps, 3),
}

B = results.get("prefill")
if isinstance(B, dict):
    b_first = B.get("first")
    b_t0 = float(B["t0"])
    b_t1 = float(B["t1"])
    out.update(
        {
            "prefill_ttft_s": round((float(b_first) - b_t0) if b_first else float("nan"), 3),
            "prefill_total_s": round(b_t1 - b_t0, 3),
            "prefill_finished_before_decode": b_t1 < a_t1,
            "overlap_s": round(max(0.0, min(a_t1, b_t1) - max(float(a_first or a_t0), b_t0)), 3),
        }
    )
else:
    out["prefill_error"] = "prefill thread did not finish"

print(json.dumps(out, indent=2, sort_keys=True))
