# Tony NVFP4 server config (as found on 2026-09-03, for a fair read of COMPARISON.md)

- Repo: `GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark` (renamed from
  `GLM-5.3-Flash-NVFP4-2x-DGX-Spark-Tony`), containers `vllm_glm53`
  (head + worker), both Up at bench time
- API: `http://127.0.0.1:8888`, served id `glm-5.3-flash`
- Weights: `RedHatAI/GLM-5.3-Flash-NVFP4` (compressed-tensors NVFP4)
- Image: `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (same family as ours)
- `MAX_MODEL_LEN=262144`, `MAX_NUM_SEQS=6`, `BLOCK_SIZE=2304`
- `MOE_BACKEND=marlin`, `KV_CACHE_DTYPE=fp8_e4m3`, `KV_CACHE_MEMORY=6442450944` (6 GiB)
- Spec: DFlash2 (`/models/dflash2-draft`), `num_speculative_tokens=7`
- Bench: same harness/methodology as `bench/bench_config.sh`
  (discarded warmup, soak 3x C1, C1/C2/C4/C6 @512 tok, temp 0, thinking off).
