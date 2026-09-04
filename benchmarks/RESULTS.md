# GLM-5.3-Flash W4A16-MTP 2x DGX Spark — Benchmark results

Measured on 2026-09-03 on a 2-node 2× DGX Spark pair (head + worker), image
`radixark/vllm-glm53-flash:sm121-v11-dflash2`
(alias `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`), TP=2, DFlash2 K=7,
`max_model_len=1048576`, KV `fp8_e4m3` pinned at 9 GiB (1,335,594 tokens).
Server at `http://<HEAD_IP>:8000`. Harness in `bench/`; per-wave data for the
validated recipe in `benchmarks/final-recipe/`.

Methodology (inherited from the NVFP4 lane): discarded warmup (1st post-boot
request doesn't count), code prompts with anti-prefix-cache variation,
`temperature=0`, `enable_thinking=false`, median of 3 runs. Single-wave
measurements carry ~±15% boot-to-boot variance — only treat gains above ~5%
with repetition as real.

## Validated final recipe (`benchmarks/final-recipe/`)

`PORT=8000`, `MOE_BACKEND=marlin`, `MAX_NUM_SEQS=6`, `BLOCK_SIZE=2304`,
`MAX_NUM_BATCHED_TOKENS=8192`, `DISABLE_FLASHINFER_AUTOTUNE=1`,
`APPLY_APC_PATCH=1`, persistent caches, DFlash2 K=7. Gate 7/7, P1 correct
(`17*23=391`, capital of Japan = Tokyo).

| metric | value |
|---|---|
| soak C1 streaming median (3x512 tok) | **31.8 tok/s**, TTFT 0.35s |
| `bench_c` C1 / C2 / C4 / C6 aggregate | ~30.4 / ~44.4 / 69.2 / **81.1 tok/s** (3-run medians; the first C1 wave read 36.53, a high outlier — see COMPARISON.md) |
| DFlash2 acceptance (`/metrics`) | 0.418 (this W4A16 lane's signature) |
| total cold boot | ~8 min |
| `init engine` | 87s (down from 287s — see § boot) |
| TileLang recompiles at boot | **0** (cache hit) |

## Single-stream decode and concurrency

| config | soak C1 | C1 | C2 | C6 | C12 | acceptance |
|---|---|---|---|---|---|---|
| baseline 8192/marlin/seq6 | 34.66 | 29.5 | 50.1 | 67.5 | — | 0.423 |
| batched 16384 | 34.30 | 33.9 | 40.5 | 71.7 | — | 0.445 |
| maxseq 12 | 30.71 | 29.2 | 45.7 | 80.0 | 73.4 / 80.3 / **84.6** | 0.423 |
| **final** | 31.82 | ~30.4 | ~44.4 | **81.1** | — | 0.418 |

Conclusions: decode ~30-36 tok/s stable across configs; aggregate saturates at
~80 tok/s already at C6 (C12 never exceeds it — MoE/NCCL-bound); acceptance
pinned at 0.41-0.45 on every boot.

## MoE backend A/B

`flashinfer_cutlass` **doesn't even boot** on W4A16:
`ValueError: moe_backend='flashinfer_cutlass' is not supported for WNA16 MoE`
(only `triton/marlin/humming/flashinfer_trtllm/emulation`). Cutlass is
NVFP4-only (NVFP4 lane). **Marlin by KO**, pinned in `.env`. Marlin repeated
over 2 boots: soak 30.69 / 35.14, C6 69.4 / 66.6.

## `max_num_batched_tokens` A/B

1024-tok decode with an overlapping ~125k-tok cold prefill, 3 waves each:

| | 8192 (winner) | 16384 |
|---|---|---|
| decode under load | 11.63 / 12.28 / 11.83 (mean **11.91**) | 10.06 / 10.99 / 11.98 (mean 11.01) |
| steady cold prefill | 48.6 / 48.6 (**48.6s**) | 48.7 / 47.5 (**48.1s**) |
| cold prefill wave1 | 53.5s | 67.3s |

Smaller chunks = less head-of-line blocking on decode (+8%), steady prefill
ties (MoE/attn-bound workload, not chunk-bound). **8192 stays.**

## Prefix cache / APC (`bench/bench_prefix_reuse.py`)

Stock: **0 hits over 9,276+ queried tokens** (byte-identical resend included).
Cause: stock vLLM flags *every* eagle group because of DFlash2 and the
drafter's SWA group zeroes the hybrid min. Fix:
`docs/patch_hybrid_prefix_hit.py` (EXL3 overlay, 4/4 anchors on our image,
fail-closed), applied as a per-node generated bind-mount
(`APPLY_APC_PATCH=1`). The log proves the new path:
`eagle_group_ids=[6]` (drafter only).

With the patch (block 2304), identical resend:

| prompt | req1 TTFT | req2 TTFT | hits | hit rate |
|---|---|---|---|---|
| ~3.6k tok | 2.6s | 2.4s | 0 | 0% |
| ~7.2k tok | 5.2s | 1.8s | 4,608 | 63.7% |
| ~21.7k tok | 18.0s | 2.2s | 18,432 | 84.9% |

Rule: `hits = full_blocks − 1` (last block falls in the eagle/MTP drop).
**Threshold: shared prefix ≥ 2 full blocks (≥ 4,609 tokens).**
Block-size 4608 was tested and refuted (0 hits) — reverted to 2304 (HF pin).
Implication for agentic coding: stable content first (system + repo map);
50k+ context runs at ~95% hit rate.

## Cold prefill, ladder

Port of the EXL3 `tests/_run_cold_prefill.py` (`bench/bench_cold_prefill.py`,
`--base/--out`, per-request salt). Temp 0, `max_tokens=8`, real TTFT:

| rung | TTFT | throughput | `local_compute` |
|---|---|---|---|
| 8k | 5.3s | 1,498 tok/s | full (cold ✅) |
| 12k | 9.4s | 1,276 tok/s | full |
| 16k | 9.9s | 1,610 tok/s | full |
| 100k | 61s | 1,638 tok/s | full |
| 256k | 168s | 1,527 tok/s | full |
| 300k | 194s | 1,549 tok/s | full |
| 8k multi-turn follow-up | 5.3s → 2.3s | 3,419 tok/s effective | 4,608 hits |

Flat prefill at ~1.3-1.6k tok/s up to 300k, no single-stream OOM.
Per-turn budget: cold ≈ 0.65ms/token; with prefix, only the tail pays.

## Cold boot (evolution)

| boot | total | `init engine` | TileLang | autotune |
|---|---|---|---|---|
| initial stock | ~11-13 min | 286s | 8 compiles | ~85s, 0 configs |
| + root-cache/FlashInfer | 617s | 117s | 7 | ~50s, 0 configs |
| + `TILELANG_CACHE_DIR` (fixed layout¹) | — | 108s | 7→0² | — |
| + `--no-enable-flashinfer-autotune` | **~7.6 min** | **87s** | **0** | skipped |
| final | ~8 min | 87s | 0 | skipped |

¹ The default is `~/.tilelang/cache` — `cache/` is part of the dir, not the
namespace; seeding the wrong layout = guaranteed miss. Correct layout now
persisted on both nodes (`tilelang/0.1.12/...` + `cuda-binaries/*.cubin`).
² Deterministic keys on all 3 boots — automatic hit; any divergence
(image/GPU/kernel) recompiles instead of using a wrong `.so`.
178G shards ≈ 207s unchanged (IO-bound) — the remaining dominant bottleneck.

## Decisions that did NOT pass (recorded so nobody repeats them)

- `flashinfer_cutlass` as MoE: incompatible with W4A16 (boot ValueError).
- PIECEWISE/FULL cudagraphs: NVFP4-lane A/B showed −8.7%/+4.3% (noise) with
  P1 divergence — we stay `enforce-eager`.
- `max_num_batched_tokens=16384`: loses on decode under load, prefill ties.
- `max_num_seqs=12`: saturates at the same ~80 tok/s as C6 — stays 6 (preserves
  KV for long agent contexts).
- `--block-size 4608`: 0 prefix hits — reverted to 2304.
- Unpinned KV (profiler at `gpu_memory_utilization=0.85`): only finds 6.96 GiB
  free < 7.05 GiB for 1x1M (`ValueError`, boot fails). The 9 GiB pin is
  *more* generous than the profiler and serves 1M stable — **keep pinned**.
