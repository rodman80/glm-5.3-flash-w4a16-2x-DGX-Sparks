#!/usr/bin/env python3
"""acceptance_ratio.py — read spec-decode acceptance from /metrics.

Usage: python3 acceptance_ratio.py [base_url]
Prints drafted / accepted / ratio. Healthy DFlash2: ratio ~0.6-0.8.
A ratio near ~0.15 indicates broken aux capture / mHC contraction.
"""
import sys
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
text = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()

vals = {}
for line in text.splitlines():
    if line.startswith("#") or not line.strip():
        continue
    name, _, value = line.rpartition(" ")
    base = name.split("{")[0]
    if "spec_decode" in base and value.replace(".", "").replace("-", "").isdigit():
        vals[base] = float(value)

drafted = vals.get("vllm:spec_decode_num_draft_tokens_total")
accepted = vals.get("vllm:spec_decode_num_accepted_tokens_total")

print("spec_decode metrics found:")
for k in sorted(vals):
    print(f"  {k} = {vals[k]}")
if drafted is None or accepted is None:
    print("ERROR: spec_decode draft/accepted counters not found")
    sys.exit(1)
ratio = accepted / drafted if drafted else float("nan")
print(f"\ndrafted={drafted:.0f} accepted={accepted:.0f} ratio={ratio:.3f}")
if 0.6 <= ratio <= 0.8:
    print("HEALTHY (in the 0.6-0.8 band)")
elif ratio < 0.3:
    print("UNHEALTHY: ratio near 0.15-class -> aux capture / mHC contraction suspect")
else:
    print("check band manually")
