#!/usr/bin/env python3
"""Medium/long-context correctness probe (temperature=0, thinking off).

Builds a ~N-token repetitive prompt (default ~100k tokens), asks for a
one-sentence summary plus 17*23, prints wall time + usage + output, saves to
--out. Run on both backends and diff for A/B correctness.
"""
import argparse
import json
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--base", default="http://10.100.24.2:8000/v1")
p.add_argument("--model", default="glm-5.3-flash")
p.add_argument("--kb-chars", type=int, default=500,
               help="prompt size in K chars (~4 chars/token)")
p.add_argument("--max-tokens", type=int, default=64)
p.add_argument("--out", default="")
a = p.parse_args()

para = ("The quick brown fox jumps over the lazy dog. Sparse attention selects top-k "
        "pools per query token; the tail carries recent context. " * 40 + "\n")
prompt = ("Summarize the following in one sentence, then answer 17*23 on its own line.\n\n"
          + para * max(1, (a.kb_chars * 1024) // len(para)))
print("prompt chars:", len(prompt))
payload = {"model": a.model,
           "messages": [{"role": "user", "content": prompt}],
           "temperature": 0, "max_tokens": a.max_tokens, "stream": False,
           "chat_template_kwargs": {"enable_thinking": False}}
t0 = time.perf_counter()
req = urllib.request.Request(a.base.rstrip("/") + "/chat/completions",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"},
                             method="POST")
with urllib.request.urlopen(req, timeout=3600) as r:
    d = json.load(r)
t1 = time.perf_counter()
c = d["choices"][0]["message"].get("content")
u = d.get("usage", {})
print("wall_s:", round(t1 - t0, 2), "usage:", u)
print("output:", repr(c))
if a.out:
    open(a.out, "w").write(f"wall={t1 - t0:.2f} usage={u}\n{c}\n")
