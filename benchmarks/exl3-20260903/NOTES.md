# EXL3 server config (as found on 2026-09-03, for a fair read of COMPARISON.md)

- Repo: `GLM-5.3-Flash-EXL3-2x-DGX-Sparks` (MiaAI-Lab), containers
  `glm53-exl3-head` / `glm53-exl3-worker`, both Up at bench time
- API: `http://127.0.0.1:8000`, served id `GLM-5.3-Flash-EXL3`
- Weights: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (rev 25a44fdb, ~164 GiB)
- Image: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`
- `MAX_MODEL_LEN=900000`, `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=7168`
- `KV_CACHE_DTYPE=fp8`, `ENFORCE_EAGER=0` (cudagraphs on)
- Spec: DFlash2 (`incoai/GLM-5.3-Flash-DFlash2`), `DFLASH_TOKENS=7`, `DFLASH_DRAFT_TP=2`
- Bench: same harness/methodology as `bench/bench_config.sh`
  (discarded warmup, soak 3x C1, C1/C2/C4/C6 @512 tok, temp 0, thinking off),
  plus C4 (= their max_num_seqs).
