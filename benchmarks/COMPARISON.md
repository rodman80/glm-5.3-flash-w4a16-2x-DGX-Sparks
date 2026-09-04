# W4A16 vs EXL3 vs NVFP4 — head-to-head on 2x DGX Spark (TP=2, DFlash2 K=7)

Same nodes, same drafter, same harness (`bench/bench_config.sh` methodology:
discarded warmup, soak 3x C1, concurrency waves @512 tok, temp 0, thinking
off, P1 determinism). W4A16 = `benchmarks/final-recipe/`; EXL3 =
`benchmarks/exl3-20260903/` (`NOTES.md` there); NVFP4 =
`benchmarks/nvfp4-tony-20260903/` (`NOTES.md` there). All runs 2026-09-03.
Run counts in parentheses — medians where n≥3.

| metric | NVFP4 (Tony) | EXL3 (Mia) | W4A16 (this recipe) | verdict |
|---|---|---|---|---|
| soak C1 streaming median | 30.63 (3-run) | 29.94 (3-run) | **31.82** (3-run) | 3-way tie ~30-32 |
| C1 aggregate | ~31.1 (32.56/29.54, n=2) | 28.33 (n=1) | ~30.4 (36.53/30.44/30.11, n=3) | tie (first 36.53 was a high outlier) |
| C2 aggregate | ~44.4 (47.39/44.96/41.87/43.83, n=4) | 31.78 (n=1) | ~44.4 (41.13/44.80/44.42, n=3) | NVFP4/W4A16 tie; EXL3 behind |
| C4 aggregate | 42.45 (n=1) | 44.43 (n=1) | **69.17** (66.30/69.17/74.46, n=3) | **W4A16 +56-63%** |
| C6 aggregate | 50.98 (n=1) | 47.87 (n=1) | **81.13** (80.08/81.13/84.14, n=3) | **W4A16 +59-69%** |
| DFlash2 acceptance | 0.423 | 0.410 | 0.418 | parity |
| P1 temp=0 | 391 / Tokyo ✓ | 391 / Tokyo ✓ | 391 / Tokyo ✓ | tie |

Reading: single-stream is a tie (~30 tok/s everywhere — acceptance also
ties ~0.41-0.42, so the spec contributes equally). The difference is
concurrency scaling: NVFP4 and EXL3 both plateau near ~45-51 from C2/C4 on,
while W4A16 keeps scaling through C4 (69.2) to C6 (81.1). The W4A16 advantage
is a high-concurrency story, not a single-stream one.

Caveats (honest): rival C4/C6 cells are single waves (±15% boot-to-boot) —
the gaps (+56-69%) are far above noise, but repeating them needs another
server swap each. Rival C1/C2 have 1-4 runs as shown; W4A16 cells are
3-4-run medians. Configs differ as each repo prescribes (NVFP4: 262K ctx,
seqs 6, KV pin 6 GiB; EXL3: 900K ctx, seqs 4, batched 7168, cudagraphs;
W4A16: 1M ctx, seqs 6, batched 8192, eager). EXL3 `max_num_seqs=4` means C6
queues on their side — still what a user experiences, but noted.
