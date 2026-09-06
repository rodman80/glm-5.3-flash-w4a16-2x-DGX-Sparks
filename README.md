# GLM-5.3-Flash W4A16 + DFlash2 on 2× DGX Spark (GB10 / SM121)

> ⚠️ **Disclaimer — work in progress.** This is my first serving recipe, built
> and validated on a single 2× DGX Spark kit. It works here (see the measured
> evidence below), but it is **not yet production-hardened**: expect rough
> edges, untested configs, and breaking changes. Standing on the shoulders of
> giants — most of the hard problems (SM121 image, top-k fix, fabric
> know-how) were solved by
> [**tonyd2wild**](https://github.com/tonyd2wild) and the sparse-MLA/prefix-cache
> groundwork by [**MiaAI-Lab**](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
> — full credits at the bottom. Thank you both!

OpenAI-compatible serving of **GLM-5.3-Flash** as the
**W4A16 INT4 + BF16 MTP** quant
([`canada-quant/glm-5.3-w4a16-mtp`](https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp),
base [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash))
on **two NVIDIA DGX Spark (GB10, SM121)** at tensor-parallel 2 — with the
[`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
speculative drafter, **fp8 KV cache**, and up to **1M context**.

As far as we can tell this is the **first W4A16 recipe for GLM-5.3-Flash on DGX
Spark** — and on our kit it is also the **fastest**: see
[benchmarks/RESULTS.md](benchmarks/RESULTS.md) for the full A/B evidence.

## Why this quant

| | BF16 | This recipe (W4A16) |
|---|---|---|
| Weights | ~599 GiB | **177.7 GiB (−70%)** |
| Serves 1M context on | 8× 80 GB | **2× DGX Spark** (also 4× H100/H200, 4× RTX PRO 6000) |
| MoE GEMMs | BF16 | INT4 group-128 GPTQ (36 288 routed-expert GEMMs only) |
| Quality-sensitive parts | — | stay BF16: attention, router/gate, shared experts, dense layers 0–2, embeddings, `lm_head`, norms, full vision tower, MTP layer |

Measured head-to-head on our 2× Spark kit, same harness (see
[benchmarks/COMPARISON.md](benchmarks/COMPARISON.md)): single-stream is a
tie (~30 tok/s everywhere, acceptance ~0.41 too) — the W4A16 advantage is
concurrency scaling: **C4 69.2 / C6 81.1 tok/s vs ~42/51 (NVFP4) and
~44/48 (EXL3)**, i.e. up to +56-69% where rivals plateau.

Quality (checkpoint card): AIME 2026 **85.0%** (102/120), GSM8K **0.97**
parity, GPQA **0.8586**.

## Final recipe performance

Validated config: `MOE_BACKEND=marlin`, `MAX_NUM_SEQS=6`, `BLOCK_SIZE=2304`,
`MAX_NUM_BATCHED_TOKENS=8192`, DFlash2 `K=7`, KV `fp8_e4m3` pinned at 9 GiB,
`--enforce-eager` ([`benchmarks/final-recipe/`](benchmarks/final-recipe/)):

| Metric | Value |
|---|---|
| Soak C1 streaming median (3× 512 tok) | **31.8 tok/s**, TTFT 0.35 s |
| `bench_c` C1 / C2 / C4 / C6 aggregate | ~30.4 / ~44.4 / 69.2 / **81.1 tok/s** (3-run medians; per-wave data in `final-recipe/`) |
| DFlash2 acceptance (`/metrics`) | 0.418 |
| Cold boot | ~8 min (`init engine` 87 s, 0 TileLang recompiles) |
| Cold prefill | ~1.3–1.6k tok/s flat up to 300K, no OOM single-stream |

Full methodology, A/B decisions (MoE backend, batched-tokens, max-num-seqs,
block-size, KV pin, prefix cache):
**[benchmarks/RESULTS.md](benchmarks/RESULTS.md)**.

## What's in this repo

| File | Role |
|---|---|
| `launch-glm53-w4a16-tp2-dflash2.sh` | TP=2 launcher, rank `0\|1` (DFlash2 spec-decode) |
| `start.sh` | 2-node orchestrator: pull → download → rsync → launch → health (`start\|restart\|stop\|status\|logs\|download\|validate`) |
| `stop.sh` / `status.sh` | thin wrappers over `start.sh` |
| `download.sh` | fetches weights (~178 GiB) + drafter (~2.3 GiB) from HF |
| `validate.sh` | pre-boot gates: HF params, shards, drafter, image, fabric GIDs, swappiness, disk, conflicting containers |
| `docker-compose.yml` | alternative head/worker compose path |
| `.env.example` | all knobs (copy to `.env` and adapt IPs/paths) |
| `PARAMS.md` | verbatim HF checkpoint parameters |
| `chat_template_mm.jinja` | vision chat template (not shipped on the HF repo — see Credits) |
| `clocks.sh` | trava os clocks GB10 em 2400 MHz nos 2 nos (hook automatico no `start.sh`) |
| `bench/` | benchmark harness (`bench_config.sh`, decode/concurrency/prefill/prefix/acceptance probes) |
| `benchmarks/` | curated results + `RESULTS.md` + `COMPARISON.md` (3-way head-to-head vs EXL3/NVFP4); per-wave data for the validated recipe in `final-recipe/` |
| `docs/` | SM121 top-k patch, hybrid prefix-cache patch + generator, SM120 sparse-MLA investigation + overlay generator |

## Performance tuning

Alem da receita default, estes ganhos foram medidos A/B no nosso par (detalhes e
becos-sem-saida em `benchmarks/perf-exp/EXP-LOG.md`):

- **Clocks GB10 em 2400 MHz** (`./clocks.sh`, hook no `start.sh`): +5–7% decode,
  +2–3% prefill, sem calor excessivo. Reset p/ stock: `./clocks.sh reset`.
- **`--async-scheduling`**: +2–3% decode, sem regressoes. Default off na build.
- **Micro opts** (`VLLM_MARLIN_USE_ATOMIC_ADD=1`, `VLLM_USE_FUSED_MOE_GROUPED_TOPK=1`):
  efeito ~0, ativas e inofensivas.
- Combinado honesto: **~+8–11% decode C1, +3% C6, +2% prefill**, sem regressoes.
- **Nao usar**: `--enable-expert-parallel` (−6% prefill, −3% C6),
  cudagraphs (neutro/pior sob carga), draft com KV bf16 (derruba acceptance
  0.42→0.33), blocos 1152/4608 (quebram prefix hits), chunks ≠ 8192.
- Teto estrutural: quant INT4-GPTQ nao usa os fast paths Blackwell (FP4); MLA
  roda FA2 em compat; acceptance DFlash2 ~0.42 fixa. +15% single-metric nao
  foi atingido com serving flags — vereditos por experimento em EXP-LOG.md.

## Requirements

- 2× DGX Spark (GB10, 128 GiB UMA each) with CX7 RoCE between them
- Docker with GPU access on both nodes, passwordless SSH head → worker
- ~190 GiB free in `/var/tmp` on both nodes (weights + drafter)
- `huggingface-cli` (or `hf`) for the download step
- `vm.swappiness=10` (or 0) on both nodes — `validate.sh` / `start.sh` check and try to fix

## Quickstart

```bash
git clone <this-repo> && cd glm-5.3-flash-w4a16-2x-DGX-Sparks
cp .env.example .env   # set HEAD_IP / WORKER_IP / SSH to YOUR pair

./validate.sh          # pre-boot gates (GIDs, swappiness, disk, image, containers)
./download.sh          # weights (~178 GiB) + drafter (~2.3 GiB), head only
./start.sh             # pull both nodes, rsync to worker, launch TP=2, poll /health
# ./start.sh restart | ./start.sh stop | ./start.sh status | ./start.sh logs [worker]
```

Manual path (if you skip the orchestrator) — **worker first**:

```bash
./launch-glm53-w4a16-tp2-dflash2.sh 1   # worker
sleep 25
./launch-glm53-w4a16-tp2-dflash2.sh 0   # head — serves :8000

until curl -sf http://<HEAD_IP>:8000/health >/dev/null; do sleep 20; done
```

Smoke test:

```bash
curl http://<HEAD_IP>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": "Prove there are infinitely many primes."}],
  "max_tokens": 128,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Or via compose (head/worker profiles): `docker compose --profile head up` /
`docker compose --profile worker up`.

## Configuration reference

Image: `radixark/vllm-glm53-flash:sm121-v11-dflash2`
(alias `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`) — vLLM dev with
NoPE-MLA fixes + FlashInfer 0.6.18. Public, no login required.

| Env | 1M (full) | 262K (staging) |
|---|---|---|
| `MAX_MODEL_LEN` | 1048576 | 262144 |
| `KV_CACHE_MEMORY` | 9663676416 (9 GiB → ~1.34M-token pool, measured 1,335,594) | 3221225472 (3 GiB) |
| `--max-num-seqs` | 6 | 6 |
| `--block-size` | 2304 | 2304 |
| `--kv-cache-dtype` | fp8_e4m3 | fp8_e4m3 |
| `--speculative-config` | `{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7}` | same |
| `--enforce-eager` | 1 | 1 |
| `--gpu-memory-utilization` | 0.85 | 0.85 |

Notes that will save you a boot cycle:

- **DFlash2 needs exactly 7 speculative tokens** — any other count wedges the boot.
- **MoE backend must be `marlin`** (or auto) — `flashinfer_cutlass` does not boot
  W4A16 (`ValueError`, NVFP4-only).
- **fp8 KV is mandatory on Spark** — bf16 has no sparse-MLA kernel on SM121.
- **Keep the 9 GiB KV pin**: without it the profiler finds only ~6.96 GiB free at
  util 0.85 and the 1M boot fails (`ValueError`). If a >3K prefill ever dies,
  empty `KV_CACHE_MEMORY` and let the profiler size it.
- **NCCL over the switch** (line-rate ~98 Gbps here); use the RoCE GID index whose
  entry is `::ffff:<your-IP>` (`cat /sys/class/infiniband/<ib>/ports/1/gids/*`).
- Poll `/health`, never `/v1/models` (returns 200 with a dead engine).
- GB10 ritual every boot on both nodes: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`.
- Single prompts ≲ ~310K tokens (larger hung the host in our tests).

## Experimental: SM120 sparse-MLA overlay

`GLM53_SM121_MLA=1` enables a feature-gated `FLASHINFER_MLA_SPARSE_SM120`
overlay (NoPE 512 → 576-wide kernel via zero-pad, exact math, Marlin/W4A16
untouched); `=0` (default) is the bit-identical SM90 baseline. A/B verdict
(2026-09-03): decode parity (±1%), correctness byte-identical, **−21% KV pool**
— keep `=0`; the overlay ships as a versioned option for future FlashInfer/SM120
work. Bring-up/A-B runbook:
[`docs/sm121-mla-investigation.md`](docs/sm121-mla-investigation.md).

## Credits

Recipe stands on the shoulders of (thank you!):

- [**zai-org/GLM-5.3-Flash**](https://huggingface.co/zai-org/GLM-5.3-Flash) — base model.
- [**canada-quant/glm-5.3-w4a16-mtp**](https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp) —
  the W4A16 quant + `recipe.yaml`/pass criteria this recipe follows verbatim.
- [**tonyd2wild**](https://github.com/tonyd2wild) /
  [`GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) —
  SM121 images (`radixark/vllm-glm53-flash:sm121-v11-dflash2`), the SM121 top-k
  fix (`docs/sparse_attn_indexer_kpool_sm121.py`), the vision
  `chat_template_mm.jinja`, and the GB10 fabric know-how (KV ladder, swappiness ritual).
- [**incoai/GLM-5.3-Flash-DFlash2**](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) —
  the DFlash2 drafter (+40–50% single-stream on GB10).
- [**MiaAI-Lab / GLM-5.3-Flash-EXL3-2x-DGX-Sparks**](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) —
  reference for the SM120 sparse-MLA approach and the hybrid prefix-cache patch
  ported here (`docs/patch_hybrid_prefix_hit.py`, `docs/patch_sm121_mla.py`).
- **RedHatAI NVFP4 lane** — benchmark methodology this harness was ported from.
- **vLLM** and **FlashInfer** upstream projects.

## License

Recipe: MIT — see [LICENSE](LICENSE). Weights, drafter and images follow their
own licenses (see LICENSE notes). Community recipe, not affiliated with
Zhipu AI, NVIDIA, or any upstream project.
