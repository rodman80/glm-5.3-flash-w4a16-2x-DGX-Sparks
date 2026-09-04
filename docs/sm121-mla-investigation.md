# SM121 sparse MLA (FLASHINFER_MLA_SPARSE_SM120) — investigation + overlay

Status: **implemented, feature-gated, A/B complete — verdict: keep SM90**
(§10, `GLM53_SM121_MLA=0`). Baseline 100% preserved; the SM120 path is
live-validated (byte-identical P1) but brings no decode gain and costs
−21% KV pool (§8b).

## 1. Current-tree diagnosis (§2)

Measured on 2026-09-03 on the head node, image
`radixark/vllm-glm53-flash:sm121-v11-dflash2`:

| Item | Value |
|---|---|
| GPU | NVIDIA GB10, compute capability **12.1** (`nvidia-smi` + `torch.cuda`) |
| torch / CUDA | 2.13.0+cu130, `torch.version.cuda` 13.0, `TORCH_CUDA_ARCH_LIST=12.1a` |
| vLLM | **0.1.dev20051+g487ecf187** (commit `487ecf187` — same as the EXL3 base) |
| FlashInfer | 0.6.18.dev20260819, `has_flashinfer_sparse_mla_sm120() = True` |
| Model | `canada-quant/glm-5.3-w4a16-mtp` (`Glm5NextForConditionalGeneration`) |
| MLA geom. | `kv_lora_rank=512`, `qk_nope_head_dim=256`, **`qk_rope_head_dim=0`** (NoPE), `v_head_dim=256` |
| Indexer | `index_topk=2048`, `index_kpool=4` |
| Quant / MoE | compressed-tensors W4A16 (`pack-quantized`, group 128) + **`'MARLIN'`** (logs: `Using 'MARLIN' WNA16 MoE backend`, `Using MarlinExperts`) |
| Current MLA backend | **`FLASHINFER_MLA_SPARSE_SM90`** (`cuda.py:533`, candidates `[SM90, SM120]`) |
| Requested → effective KV | `fp8_e4m3` → `fp8_e4m3` (SM90 doesn't canonicalize; kernel logs `kv_cache_dtype=torch.float8_e4m3fn`) |
| block_size | 2304 requested → **4608** after mamba hybrid align (`interface.py:926`) |
| Effective Q in kernel | absorbed 512-wide (`mqa_ql_nope [B,N,512]` = `q_nope @ W_UK^T`), `q_pe` width 0 |
| major==12 selector | `[TRITON_MLA, SM90, SM120]`; `supports_compute_capability = major == 12` → **SM121 eligible** |
| SM120 canonicalization | **already exists** (`mla_attention.py:354`: `auto/fp8/fp8_e4m3 → fp8_ds_mla` for SM120) |
| PDL | already gated `major in (9, 10)` in this tree (no patch) |
| `PAGED_MQA_PAGE_SIZES` | `(32, 64)` arch-blind (cuda.py patch target) |

Required confirmations: (1) weights stay W4A16 — overlay doesn't touch
quantization/MoE; (2) Marlin stays pinned (`MOE_BACKEND=marlin`);
(3) change restricted to attention/KV (6 files, all outside quant);
(4) capability 12.1 passes the SM120 selector (`major == 12`).

## 2. Comparison with the references

**Primary (EXL3, working):** `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`,
Dockerfile "GLM-5.3-Flash NoPE sparse MLA on SM121" + README ("Why the
overlay exists", "KV cache"). Runs `FLASHINFER_MLA_SPARSE_SM120` with
`--kv-cache-dtype fp8 → fp8_ds_mla` on GB10/SM121. Our tree is the **same
commit** (`487ecf187`): **all 14 Dockerfile anchors match byte-exact**
(verified by `docs/patch_sm121_mla.py`, fail-closed).

**Secondary (NVFP4, historical):** the old NVFP4 repo's
`files/glm53-flash_SM121.py` (file removed when that repo was renamed to
`GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark`) marks the
sm120 packed path as *"unused / do not bake"*. Differences explaining why it
never worked and EXL3 did: (a) global patch in `backend.py::do_kv_cache_update`
(affects every backend) vs local override in the SM120 impl; (b) Q-pad without
`rope_pad`/validation; (c) top-k by **blind truncate/pad to 2048** — drops the
recent tail (highest attention weight!) — vs dropping the lowest-ranked pool;
(d) no `supports_dense_mha_prefill=False`, no `buffer_width`, no `[64]` pin.
We use nothing from it; it stands only as evolution record.

**Upstream:** `flashinfer_mla_sparse.py` (SM120: `auto/fp8/fp8_e4m3/fp8_ds_mla`,
`major == 12`), `_canonicalize_sparse_mla_kv_cache_dtype` (SM120 canonicalizes),
`concat_and_cache_mla` (`pe_dim == 64` for `fp8_ds_mla`), issue **#53963**
(3 failure modes: `pe_dim must be 64`; bf16 with no backend on SM12x;
`kv_lora==512 && qk_rope==64 && query==576` guard in
`flashinfer/mla/_core.py:582` — confirmed in the installed FlashInfer).

## 3. Incompatibility gap (precise)

Not semantic — **interface geometry/layout**: NoPE `(512, 0, 512)` vs the
GLM_NSA kernel `(512, 64, 576)`. Exact representation via zero-pad
(`Q'·K' = Q·K`, proven in `bench/sm121_mla_math_test.py`), values coming from
the 512 region. Three sub-gaps + fixes (all in the overlay):
1. **KV write** — `pe_dim=0` rejected → synthetic zeros(64) `k_pe` in the impl.
2. **Query/kernel** — `_core.py:582` guard → Q 512→576 + `qk_rope=64` only at
   the kernel boundary (model stays NoPE).
3. **Top-k template** — 2176 buffer vs mandatory `topk==2048` (decode
   dispatch only instantiates {128,512,1024,2048}) → 2048 buffer + lowest-rank
   pool trim (511·4+3 = 2047 ≤ 2048, `select_k=512` keeps the fast path) +
   `valid_counts/capacity/empty_rows` in the impl (port of the SM100 sibling).
4. **Pages** — SM120 `[64,256]→[64]` (kernels only at 64); indexer align
   `index_kpool·64=256` on SM12 (DeepGEMM sm_120 only takes `block_kv=64`). Our
   `--block-size 2304` is already a multiple of 256 and the 576 storage is a
   multiple of 64 — no conflict.

## 4. Implemented patches (§14.4) + flag (§14.5)

- `docs/patch_sm121_mla.py` — fail-closed/idempotent generator (extracts
  pristine from `$IMAGE`, applies, verifies `compile()` + markers; any
  divergence aborts without writing). Includes §9 asserts (`kv_lora==512`,
  `index_topk==2048`, `major==12`, `Q==512`, `K_PE==0`, `kernel_rope==64`) and
  unique `[SM121 MLA] ...` logs.
- `launch-glm53-w4a16-tp2-dflash2.sh` — `GLM53_SM121_MLA=1` mounts 6 files + pins
  **only the `mla_attention` group** via
  `--attention-config '{"backend_per_kind": {"mla_attention": "FLASHINFER_MLA_SPARSE_SM120"}}'`
  (parse validated against the image; indexer/KDA/vision stay auto — a global
  `--attention-backend` pin would break non-MLA layers). DFlash2 drafter:
  `kv_cache_dtype:auto` in SPEC_JSON, since the target canonicalizes the
  global cache to `fp8_ds_mla` and no non-MLA backend accepts it (without this
  the drafter dies with `ValueError: No valid attention backend` — observed in
  the 2026-09-03 bring-up). `=0`: bit-identical baseline (dry-run validated).
  Generation failure **aborts** (exit 3).
- `.env`/`.env.example` — `GLM53_SM121_MLA=0` default.
- `docker-compose.yml` out of scope: static path without generator; A/B uses
  `./start.sh` + launch script.
- `bench/sm121_mla_math_test.py` — QK/softmax identity (50 trials),
  656 B geometry (+24.2%), indexer arithmetic; `SM121_VERIFY_OVERLAY=1`
  re-verifies the generated overlay.
- **NOT ported (test-first):** sparse FlashInfer warmup/autotune skip and
  fused_moe autotuner (current boot already uses
  `--no-enable-flashinfer-autotune` and doesn't hang; apply only if the SM120
  bring-up stalls) and no PDL (already gated).

## 5. Composite indexer base

The current `PATCH_KPOOL_HOST` mount (`persistent_topk` gate for SM<78) is used
as the **base** of the flag=1 indexer (`--base-indexer`), so flag=1 ⊃ flag=0.
Without it, start from pristine.

## 6. Ordered bring-up (§11) — runbook

Prereqs: normal `./validate.sh` + `./start.sh`; do **not** tear down the
current server to generate the overlay (generation is offline, only reads the
image).

- **Step 1 (eager, 1 req):** 262K staging (`MAX_MODEL_LEN=262144`,
  `KV_CACHE_MEMORY=3221225472`, no 9 GiB pin), `MAX_NUM_SEQS=1`,
  `SPEC_METHOD=mtp` + `MTP_TOKENS=2` (isolates the DFlash2 drafter),
  `GLM53_SM121_MLA=1`, alternate port (`PORT=8001`) if prod is up.
  Wait for `/health` (~6-8 min cold). Check §7 logs.
- **Step 2 (correctness):** `temperature=0`, same prompt/seed, baseline vs
  SM120: first tokens, full output, NaN/Inf, short and long context
  (~256K), prefix caching on/off, multi-seq, `block_tables`/`index_topk`.
- **Step 3 (real workload):** re-enable 1M + 9 GiB pin + `MAX_NUM_SEQS=6` +
  `MAX_NUM_BATCHED_TOKENS=8192` + DFlash2 K=7.
- **Step 4 (spec):** only after the target validates; if it stalls in
  warmup/autotune (rank 0, PDL/KDA race), port the GB10 block from the EXL3
  Dockerfile.

## 7. Expected success evidence (§15)

```
Using FLASHINFER_MLA_SPARSE_SM120 attention backend ...        # selector (mla_attention kind)
Using fp8_ds_mla KV cache format for FLASHINFER_MLA_SPARSE_SM120 backend.  # canonicalization
[SM121 MLA] GLM-5.3 NoPE compatibility mode enabled ...        # 1x per worker
[SM121 MLA] Q 512 -> 576 zero padded                           # 1x
[SM121 MLA] K_PE 0 -> 64 synthetic zeros                       # 1x
Using 'MARLIN' WNA16 MoE backend. + Using MarlinExperts        # weights intact
GPU KV cache size: ~1.0-1.1M tokens (1M+pin)                   # ~24% below baseline 1.335M
```

## 8. Validation done (without tearing down prod)

- [x] 14/14 byte-exact anchors; overlay generates + `compile()` + verify OK;
      idempotent; `--verify-only` OK
- [x] `bench/sm121_mla_math_test.py` PASS (QK, 656 B, indexer)
- [x] `bash -n` launch; flag=1 dry-run (6 mounts + pin, no dup) and flag=0
      (identical to current); `--attention-config` parse validated on image
- [x] **Live SM120 boot OK (2026-09-03, `SPEC_METHOD=none`, 1M+9 GiB pin)**:
      `FLASHINFER_MLA_SPARSE_SM120` selected (`mla_attention` kind),
      `fp8_ds_mla` canonicalized, `[SM121 MLA] NoPE compatibility` +
      `Q 512->576` + `K_PE 0->64` 1x at init and 1x at first forward,
      Marlin intact, **1,221,126-token** pool (SM90 baseline: 1,335,594,
      −8.6%), P1 temperature=0 **byte-identical** to baseline, no NaN/crash
- [x] `bench_config.sh sm120-mla-nospec` battery + A/B table (done — §8b)
- [ ] DFlash bring-up (Step 4, blocked on the glm5n fix §9b) and spec A/B

## 8b. Clean kernel A/B (2026-09-03, evening)

Same `bench_config.sh` harness, same 1M+9 GiB pin config, no spec on either
arm (the `final-recipe` DFlash arm is reference only):

| metric | SM90 no-spec (`sm90-nospec`) | SM120 no-spec (`sm120-mla-nospec`) | SM120/SM90 |
|---|---|---|---|
| soak C1 decode med | 14.205 tok/s | 14.303 tok/s | **1.007×** |
| c1 aggregate | 14.13 | 14.22 | 1.006× |
| c2 aggregate | 27.33 | 27.30 | 0.999× |
| c6 aggregate | 54.68 | 54.81 | 1.002× |
| P1 temp=0 | `17×23=391 / Tokyo` | **byte-identical** | ✓ |
| ctx ~100K cold (prefill+33 tok) | 66.0 s | 62.2 s | ~parity |
| ctx ~100K repeated (prefix hit) | — | 4.8 s (APC hit) | prefix OK |
| KV pool | 1,544,105 tok | 1,221,126 tok | **−20.9%** |
| `final-recipe` SM90+DFlash (ref) | soak 31.8 / c1 36.5 / c2 41.1 / c6 81.1, accept 0.418, pool 1,335,594 | — | gap is spec, not kernel |

Reading: **decode parity (±1%), identical correctness, −21% capacity**.
The 31.8→14.2 gap to the DFlash arm is the spec win (accept 0.418), not the
kernel. Likely cause of no gain: the SM90 lane reads 512 B/token fp8 with
in-kernel dequant (less HBM traffic) vs 656 B packed on SM120 — in
bandwidth-bound decode, traffic wins.

## 9. Memory trade-off (§13)

DSA: 656 B vs ~528 B ideal (**+24.2%**). The 1M+pin pool should drop from
~1.336M (measured baseline) to ~1.0-1.1M tokens. Report alongside tok/s —
don't conclude on throughput alone.

## 9b. Bring-up finding: DFlash + glm5n formula (2026-09-03)

Two SM120+DFlash boots failed AFTER the target loaded (target overlay OK):
1. `ValueError: No valid attention backend` on the drafter (non-causal SWA +
   `fp8_ds_mla` inherited from the canonicalized global) → patchless fix:
   dflash SPEC_JSON with `"kv_cache_dtype":"auto"` (`SpeculativeConfig`
   field, already supported in `load_dflash_model`).
2. `ValueError: ... 46.34 GiB needed > 9.0 GiB` (1M) and `40.75 > 11.21`
   (262K). Temporary instrumentation (reverted) of the `glm5n` branch showed:
   `mla_names=11 mla_page=3022848 idx_names=11 idx_page=152064 draft=5
   draft_page=32768 blocks_needed=1247 per_block=35087872 total=43.75 GB`.
   The drafter contributes **1153 blocks** (2048 window + ~16384 inflight with
   spec, block 16) and EACH is billed at the shared 35 MB `per_block`
   (11×3.02 MB MLA + 11×152 KB idx + 5×32 KB draft) → ~40 GB. Real incremental
   cost is only ~7.6 KB/token; the rest is the formula's fixed cost, which
   assumes the EXL3 slot-share world (drafter mounted on MLA tensors). Our
   standalone-bf16 drafter would need ~184 MB, not 40 GB.
   Decision (§11: spec last): target bring-up and A/B with **MTP**
   (`SPEC_METHOD=mtp`, no draft group → ~94 blocks × 35 MB ≈ 3.3 GB, fits the
   pin). DFlash returns in Step 4 with either: (a) accounting fix on the glm5n
   branch (bill draft blocks at `draft_page`), or (b) EXL3 slot-share (out of
   scope: allocator surgery). Compare `final-recipe` (SM90+DFlash) vs
   SM120+MTP to measure the MLA-kernel delta (the hypothesis) without the spec
   win — document in the A/B table.

## 10. Recommendation: KEEP SM90

**KEEP SM90** for production. The SM120 overlay is correct (full parity) but
brings no decode gain, costs −21% pool, and adds 6 mounts + a per-group pin.
Keep `GLM53_SM121_MLA=0` with the overlay as a versioned option for future
FlashInfer (the SM120 lane may evolve) or workloads where the 2048 template
helps — re-evaluate if loaded `mixed-prefill` or >500K contexts show a
difference (not measured; optional backlog).
Pending future engineering (out of this work): glm5n accounting fix for
DFlash + SM120 (Step 4) and the native MTP loader KeyError (pre-existing,
out of scope: `model.layers.45...w2_weight`).

## 10b. Process incident (2026-09-03)

A launch dry-run with a `docker` stub delegating `rm` to the real docker
removed the head prod container. Recovered via `./start.sh restart` →
subsequent bring-ups. Lesson: dry-run stubs never delegate mutations.

## 11. Rollback note (kept from the preliminary)

If a future regression appears: `GLM53_SM121_MLA=0` + reboot (rollback = one
env). If warmup/autotune stalls under another config: port only the matching
GB10 block from EXL3.
