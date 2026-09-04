#!/usr/bin/env python3
"""bench_acceptance.py — per-run DFlash2 acceptance measurement.

Runs INSIDE the vLLM container (needs the local tokenizer under /models and
the API on 127.0.0.1:8000). For one labeled run it:

  1. snapshots the vLLM spec-decode + prefix-cache counters (/metrics) and
     the acceptance-probe state file (if the probe is active);
  2. sends one greedy (temperature=0) completion with a deterministic prompt
     of exactly --prompt-tokens tokens (seeded; same seed = same prompt, so
     re-running the same seed exercises the prefix cache);
  3. snapshots again and reports the deltas:
       - tok/s, prefix-cache hit rate (block granularity 2304 tokens)
       - accepted/drafts (tokens per target step)
       - per-position acceptance (pos0..pos6)
       - probe: in_topk / not_in_topk of the target's actual token in the
         selector's 16-candidate set at the first rejected position, plus
         target-token rank (when present) and selected-token rank (accepted).

Usage (inside the container):
  python3 bench_acceptance.py --label cold512 --seed 1 --prompt-tokens 512
  python3 bench_acceptance.py --label warm512 --seed 1 --prompt-tokens 512
  python3 bench_acceptance.py --label cold4608 --seed 2 --prompt-tokens 4608
  python3 bench_acceptance.py --label warm4608 --seed 2 --prompt-tokens 4608

4608 = 2 x 2304-token blocks -> a warm run gets a FULL prefix hit; 512-token
prompts can never hit (below one block).
"""
import argparse
import hashlib
import json
import re
import time
import urllib.request

API = "http://127.0.0.1:8000"
MODEL = "glm-5.3-flash"
PROBE_FILE = "/tmp/dflash2-acc-state.json"

# Real, diverse Python source (vLLM's own v1 tree, read from the container)
# is used as the prompt corpus so acceptance is NOT inflated by repetition.
# The corpus is loaded lazily and rotated by seed; same seed => identical
# prompt.
_CORPUS_CACHE: str | None = None


def _load_corpus() -> str:
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None:
        import glob

        files = sorted(
            glob.glob(
                "/usr/local/lib/python3.12/dist-packages/vllm/v1/**/*.py",
                recursive=True,
            )
        )[:300]
        parts = []
        total_chars = 0
        for path in files:
            try:
                with open(path, errors="ignore") as fh:
                    text = fh.read()
            except OSError:
                continue
            parts.append(text)
            total_chars += len(text)
            if total_chars > 110000:  # ~25-30k tokens, no repetition < 16K
                break
        _CORPUS_CACHE = "\n".join(parts)
    return _CORPUS_CACHE


_LEGACY_CORPUS = r'''
import math
from dataclasses import dataclass, field

@dataclass
class BlockTable:
    blocks: list[int] = field(default_factory=list)
    size: int = 2304
    def slot(self, pos: int) -> int:
        b = pos // self.size
        return self.blocks[min(b, len(self.blocks) - 1)] * self.size + pos % self.size
    def append_block(self, block_id: int) -> None:
        self.blocks.append(block_id)

class KVCache:
    """Paged KV cache with block-aligned prefix reuse."""
    def __init__(self, num_blocks: int, block_size: int, dtype: str = "fp8_e4m3"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free = list(range(num_blocks))
        self.dtype = dtype
    def allocate(self) -> int:
        return self.free.pop()
    def release(self, block_id: int) -> None:
        self.free.append(block_id)

def rms_norm(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]:
    n = len(x)
    ssq = sum(v * v for v in x) / n
    r = 1.0 / math.sqrt(ssq + eps)
    return [x[i] * r * weight[i] for i in range(n)]

def fused_moe_gate_up(x: list[float], w_gate: list[list[float]], w_up: list[list[float]]):
    hidden = len(x)
    inter = len(w_gate)
    gate = [sum(x[i] * w_gate[k][i] for i in range(hidden)) for k in range(inter)]
    up = [sum(x[i] * w_up[k][i] for i in range(hidden)) for k in range(inter)]
    return [g * (u / (1.0 + u * u)) for g, u in zip(gate, up)]

def topk_routing(logits: list[float], top_k: int) -> list[tuple[int, float]]:
    order = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
    keep = order[:top_k]
    total = sum(math.exp(logits[i]) for i in keep)
    return [(i, math.exp(logits[i]) / total) for i in keep]

static inline float rope_neox(float x, float y, float cos, float sin) {
    return x * cos - y * sin;
}

typedef struct {
    int block_id;
    int offset;
    uint64_t hash;
} KVSlot;

static int slot_lookup(KVSlot *table, int n, int pos, int block_size) {
    int b = pos / block_size;
    int o = pos % block_size;
    for (int i = 0; i < n; i++) {
        if (table[i].block_id == b) {
            return table[i].offset + o;
        }
    }
    return -1;
}

void allreduce_sum_bf16(float *buf, int n, int rank, int world) {
    for (int i = 0; i < n; i++) {
        float acc = 0.0f;
        for (int r = 0; r < world; r++) {
            acc += buf[r * n + i];
        }
        buf[rank * n + i] = acc;
    }
}

def spec_decode_step(draft: list[int], target_logits: list[list[float]]) -> int:
    """Greedy speculative decoding: count accepted draft tokens."""
    accepted = 0
    for k, tok in enumerate(draft):
        if target_logits[k][tok] == max(target_logits[k]):
            accepted += 1
        else:
            break
    return accepted

def gumbel_argmax(logits: list[float], noise: list[float]) -> int:
    best, best_i = -1e30, 0
    for i, (l, g) in enumerate(zip(logits, noise)):
        v = l + g
        if v > best:
            best, best_i = v, i
    return best_i

class PrefixCache:
    """Block-aligned prefix cache: a block only hits when fully populated."""
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.blocks: dict[int, bytes] = {}
    def store(self, prefix: bytes, key: int) -> None:
        full = len(prefix) // self.block_size
        for i in range(full):
            self.blocks[key * 1000 + i] = prefix[i * self.block_size:(i + 1) * self.block_size]
    def hit_length(self, prefix: bytes, key: int) -> int:
        i = 0
        while i * self.block_size + self.block_size <= len(prefix):
            if self.blocks.get(key * 1000 + i) != prefix[i * self.block_size:(i + 1) * self.block_size]:
                break
            i += 1
        return i * self.block_size

def chunked_prefill(tokens: list[int], chunk: int = 2048):
    for i in range(0, len(tokens), chunk):
        yield tokens[i:i + chunk]

def attention_scores(q: list[float], k: list[list[float]], scale: float) -> list[float]:
    return [sum(q[i] * row[i] for i in range(len(q))) * scale for row in k]

def softmax(x: list[float]) -> list[float]:
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]

def mlp(x: list[float], w1: list[list[float]], w2: list[list[float]]) -> list[float]:
    h = [sum(x[i] * w1[k][i] for i in range(len(x))) for k in range(len(w1))]
    h = [v * v / (1.0 + v * v) for v in h]
    return [sum(h[i] * w2[k][i] for i in range(len(h))) for k in range(len(w2))]

def kv_quantize_fp8(x: list[float], scale: float) -> list[int]:
    return [max(-127, min(127, int(v / scale))) for v in x]

def kv_dequantize_fp8(q: list[int], scale: float) -> list[float]:
    return [v * scale for v in q]

def paged_attention(q: list[float], kv_blocks: list[list[list[float]]],
                    block_table: list[int], seq_len: int, scale: float) -> list[float]:
    k = []
    v = []
    for b in block_table:
        kb, vb = kv_blocks[b]
        k.extend(kb)
        v.extend(vb)
    k = k[:seq_len]
    v = v[:seq_len]
    s = attention_scores(q, k, scale)
    p = softmax([x - 2.0 * i for i, x in enumerate(s)])
    return [sum(p[i] * v[i][j] for i in range(len(v))) for j in range(len(v[0]))]
'''


def build_prompt_ids(tokenizer, n_tokens: int, seed: int) -> list[int]:
    """Deterministic prompt of exactly n_tokens for a seed.

    Uses the real-source corpus (no repetition below ~16k tokens) so the
    drafter cannot trivially predict repeated text.
    """
    base = _load_corpus()
    if len(base) < n_tokens * 4:
        # Fallback: pad with the legacy synthetic corpus if the real corpus
        # is too short for the requested length.
        base = base * 2 + _LEGACY_CORPUS * 4
    rot = (seed * 7919) % len(base)
    text = (base[rot:] + base[:rot]) * 2
    toks = tokenizer.encode(text, add_special_tokens=False)
    return toks[:n_tokens]


def get_metrics(api: str) -> dict:
    with urllib.request.urlopen(api + "/metrics", timeout=30) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(
            r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*?position="(\d+)"[^}]*\} ([0-9.eE+]+)$',
            line,
        )
        if m:
            out[f"per_pos{m.group(1)}"] = float(m.group(2))
            continue
        m = re.match(r"^(vllm:[a-z_]+)(?:\{[^}]*\})? ([0-9.eE+]+)$", line)
        if m and m.group(1) not in out:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def read_probe(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def complete(api: str, model: str, prompt_ids: list[int], max_tokens: int):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        api + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read().decode())
    wall = time.time() - t0
    return wall, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--api", default=API)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--probe-file", default=PROBE_FILE)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "/models/glm-5.3-flash-w4a16-mtp", trust_remote_code=True
    )
    prompt_ids = build_prompt_ids(tok, a.prompt_tokens, a.seed)
    assert len(prompt_ids) == a.prompt_tokens

    m0 = get_metrics(a.api)
    p0 = read_probe(a.probe_file)
    wall, out = complete(a.api, a.model, prompt_ids, a.decode_tokens)
    m1 = get_metrics(a.api)
    p1 = read_probe(a.probe_file)

    def d(key: str) -> float:
        return m1.get(key, 0.0) - m0.get(key, 0.0)

    drafts = d("vllm:spec_decode_num_drafts_total")
    accepted = d("vllm:spec_decode_num_accepted_tokens_total")
    per_pos = [m1.get(f"per_pos{i}", 0.0) - m0.get(f"per_pos{i}", 0.0) for i in range(7)]
    p_hits = d("vllm:prefix_cache_hits_total")
    p_queries = d("vllm:prefix_cache_queries_total")

    gen_tokens = out.get("usage", {}).get("completion_tokens", 0)
    text = out["choices"][0]["text"] if out.get("choices") else ""
    result = {
        "label": a.label,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "text_tail": text[-40:].replace("\n", "\\n"),
        "seed": a.seed,
        "prompt_tokens": a.prompt_tokens,
        "decode_tokens_requested": a.decode_tokens,
        "decode_tokens": gen_tokens,
        "wall_s": round(wall, 3),
        "tok_s": round(gen_tokens / wall, 3) if wall else None,
        "prefix_hits": p_hits,
        "prefix_queries": p_queries,
        "prefix_hit_rate": round(p_hits / p_queries, 4) if p_queries else None,
        "drafts": drafts,
        "accepted": accepted,
        "acc_per_step": round(accepted / drafts, 4) if drafts else None,
        "per_pos_accepted": per_pos,
        "per_pos_rate": [
            round(per_pos[i] / drafts, 4) if drafts else None for i in range(7)
        ],
        "per_pos_cond": [
            round(per_pos[i] / per_pos[i - 1], 4)
            if i > 0 and per_pos[i - 1]
            else None
            for i in range(7)
        ],
    }
    if p0 and p1:
        c0, c1 = p0["cum"], p1["cum"]
        probe = {}
        for k in c1:
            probe[k] = c1[k] - c0[k]
        if probe.get("rejected"):
            probe["in_topk_pct"] = round(
                100.0 * probe["in_topk"] / probe["rejected"], 2
            )
            probe["not_in_topk_pct"] = round(
                100.0 * probe["not_in_topk"] / probe["rejected"], 2
            )
        if probe.get("tgt_rank_n"):
            probe["tgt_rank_avg"] = round(probe["tgt_rank_sum"] / probe["tgt_rank_n"], 2)
        if probe.get("sel_rank_n"):
            probe["sel_rank_avg"] = round(probe["sel_rank_sum"] / probe["sel_rank_n"], 2)
        result["probe"] = probe

    print(json.dumps(result, indent=2))
    pp = " ".join(f"p{i}={result['per_pos_rate'][i]}" for i in range(7))
    pr = (
        f" in_topk={result['probe'].get('in_topk_pct')}%"
        if "probe" in result
        else ""
    )
    print(
        f"[{a.label}] {gen_tokens} tok / {wall:.1f}s = "
        f"{result['tok_s']} tok/s | prefix_hit="
        f"{result['prefix_hit_rate']} | acc/step={result['acc_per_step']} "
        f"| sha={result['text_sha256']} | {pp}{pr}",
        flush=True,
    )


if __name__ == "__main__":
    main()
