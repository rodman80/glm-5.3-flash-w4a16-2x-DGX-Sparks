# Perf experiments — decode/prefill TPS (branch `perf/speedup`)

Goal: ≥15% decode **or** prefill gain vs default recipe, same model/quant
(`canada-quant/glm-5.3-w4a16-mtp` + DFlash2 drafter, TP=2, 2× DGX Spark).
Step success bar: ≥2.5% → keep; else discard. Verify: ≥3 runs each arm.

## Baseline (default recipe, server up 23h, 2026-09-06 pre-restart)

`MOE_BACKEND=marlin`, `MAX_NUM_SEQS=6`, `BLOCK_SIZE=2304`,
`MAX_NUM_BATCHED_TOKENS=8192`, `--enforce-eager`, DFlash2 K=7,
KV `fp8_e4m3` pinned 9 GiB, `MAX_MODEL_LEN=1048576`.

| metric | runs | median |
|---|---|---|
| C1 decode (bench_decode, 512 tok) | 23.28 / 29.61 / 35.33 | **29.61** |
| C6 aggregate (bench_c, 6×512) | 77.63 / 82.52 / 79.45 | **79.45** |
| cold prefill 8k/12k/16k/100k/256k/300k | single wave | 1531 / 1580 / 1611 / 1642 / 1600 / 1585 |

Raw: `benchmarks/baseline-20260906/prefill.json`. Note C1 variance ±15-20%.

Already-failed (RESULTS.md, do not repeat without new info):
marlin≫triton (2×), humming/trtllm-cutlass don't boot, batched 16384
(worse decode-under-load, prefill tie), maxseq 12 (saturates same ~80),
block 4608 (0 prefix hits), unpinned KV (boot ValueError), SM120 overlay
(decode parity, −21% KV pool).

## Exp A — cudagraphs (ENFORCE_EAGER=0, -cc cudagraph_mode=PIECEWISE)

Rationale: eager pays per-step launch overhead on 60+ MoE layers;
vLLM#49547 measured −16% for PIECEWISE-vs-FULL downgrade, implying
cudagraphs ≫ eager for spec-decode. NVFP4-lane A/B saw noise, but that
was a different quant; re-test on W4A16 properly.
Launcher: `ENFORCE_EAGER`/`CUDAGRAPH_MODE` env (image has no
`--cudagraph-mode` CLI flag → `--compilation-config '{"cudagraph_mode":…}'`).
Status: booting (restart3).

## Exp A verdict (2026-09-06): FAIL — discard, reverted to eager

C1 (9 runs pooled): 28.27–31.21 + one 40.64 outlier, median 29.74 vs
baseline 29.61 (+0.4%, noise). C6: 78.65/72.11/77.58 median 77.58 vs
79.45 (−2.3%). Prefill: 8k 1511 vs 1531, 16k 1580 vs 1611, 100k 1562 vs
1642, 256k 1427 vs 1600, 300k 1525 vs 1585 (−1~−5%, 12k rung outlier
ignored). PIECEWISE graphs confirmed in log, P1 ✓. Conclusion: MoE
step is kernel-bound (marlin M≈8 + drafter), graphs only remove launch
overhead → neutral. Also learned: per-pos conditional acceptance is
flat ~0.77 (pos1..6/prev), so tail drafts contribute a lot — K=7 stays,
do NOT cut speculative tokens.
Raw: /tmp/expA-prefill.json + /tmp/expA-prefill.log.

## Exp B — enable-expert-parallel=1 (with eager)

Rationale: TP allreduce per MoE layer on every step → EP uses
dispatch/combine instead; HF H100 recipe uses EP. Unknown sign on
RoCE-TP2 — measure C1/C6 + prefill.

## Exp B verdict (2026-09-06): MIXED — +3% decode, −7% prefill

Config: eager + `--enable-expert-parallel`, P1 ✓, EP confirmed
(`enable_expert_parallel: True`, workers TP0_EP0).
C1: 29.79 / 33.51 / 30.50 → median **30.50** vs baseline 29.61 (**+3.0%**,
above the 2.5% keep-bar but thin — needs multi-boot confirmation).
C6: 76.16 / 78.92 / 79.94 → median 78.92 vs 79.45 (−0.7%, tie).
Prefill: 8k 1408 vs 1531 (−8%), 16k 1334 vs 1611 (−17%),
100k 1541 vs 1642 (−6%), 256k 1498 vs 1600 (−6%), 300k 1480 vs 1585
(−7%). Consistent ~−7% prefill regression (EP dispatch/combine hurts
large-M). Raw: /tmp/expB-prefill.json/.log.
Utilization (this boot): SM ~94-96% at BOTH C1 and C6, power only ~28W
→ kernel/UMA-bandwidth-bound, not launch-bound (consistent with Exp A
neutral). Decode stacking needs less-work-per-step, not less-overhead.
Decision: revert EP for now (prefill path has more headroom);
EP decode +3% stays in pocket for decode-path stacking later.

## Exp C — mamba_cache_mode=all (EP reverted to 0)

Rationale: default 'align' may recompute mamba states across chunked-
prefill chunks; 'all' caches full state → possible prefill win.
Gate: boot success + KV pool size (expect ~1,335,594) + P1.
References: prefill vs baseline-20260906 (EP0/align), decode vs same.

## Exp C verdict (2026-09-06): FAIL — discard, reverted to 'align'

Config: EP0 + `--mamba-cache-mode all`. Boot OK, KV pool intact
(1,335,594 — no memory cost), P1 ✓, `mamba_cache_mode: 'all'` confirmed.
C1: 33.32 / 30.11 / 30.42 → pooled median 30.42 vs 29.61 (+2.7%, within
run-to-run noise; first-set 33.3 was post-boot luck).
C6: 77.28 / 83.31 / 76.43 → median 77.28 vs 79.45 (−2.7%).
Prefill: 8k 1542 (+0.7%), 16k 1610 (+0%), 100k 1619 (−1.4%),
256k 1504 (−6%), 300k 1579 (−0.3%). 12k rung flaky 3rd boot in a row
(8.8s — systematic harness/engine quirk, ignore).
Conclusion: mamba 'all' changes nothing measurable. Screening lesson:
C1 run-to-run noise is ±10%; use C6-median + long prefill rungs for
screening (stable), full 3×3 only at verify.
Raw: /tmp/expC-prefill.json/.log.

## Exp D — SPEC_METHOD=mtp, MTP_TOKENS=4 (EP0, align, eager)

Rationale: DFlash2 draft = full extra model forward per step; MTP heads
are integrated (tiny draft cost). Tony's NVFP4 MTP-4 did 21.8 vs DFlash
46.9 — but W4A16+MTP is untested on this lane, and W4A16 DFlash
acceptance may be quant-degraded while MTP (same backbone) is not.
If MTP step ≪ DFlash step, MTP can win despite fewer accepts.
Gate: boot + P1 + acceptance counters (MTP reports spec counters too).

## Exp D verdict (2026-09-06): FAIL — incompatible, reverted to dflash

`SPEC_METHOD=mtp` boot dies in MTP draft loader:
`KeyError: 'model.layers.45.mtp_block.mlp.experts.routed_experts.w2_weight'`.
Root cause: the canada-quant checkpoint ships **zero MTP weights**
(scanned all 11 shards, 0 keys matching 'mtp') — there is no MTP layer
to draft from. (The "MTP" in the quant name is training lineage, not a
shipped layer.) MTP spec is untestable on this quant. Reverted.
Note: expert naming also differs (`mlp.experts.N.*` vs loader's
`routed_experts`), so even a future +MTP checkpoint would need a loader
mapping patch.

## Exp E — drafter TP=1 (DRAFT_TP=1, dflash K=7, EP0, align, eager)

Rationale: `draft_tensor_parallel_size` supports 1 or target-TP.
TP1 = full 2.2GB drafter replica per rank, zero cross-node NCCL in the
draft forward (a small dense Qwen3 whose TP2 forward is likely
latency/NCCL-bound). vLLM builds a separate draft_parallel_config and
redistributes draft states to the TP2 target. If draft is ~30% of step
time, this can be a large decode win with identical acceptance.
Gate: boot + P1 + acceptance ratio (must stay ~0.42) + prefill (draft
prefill also goes TP1 — watch large rungs).

## Exp E verdict (2026-09-06): WEAK KEEP (C6 +2.8%, needs verify)

Config: dflash K=7 + `draft_tensor_parallel_size=1`, EP0, align, eager.
Boot OK, P1 ✓, acceptance healthy (0.57 tiny-sample → per-window
mean-length 2.7–5.0 during benches, same character as baseline).
C1 (9 runs): medians 29.64/29.51/29.58 → **tie** with baseline 29.61.
C6: 79.05 / 85.00 / 81.70 → median **81.70** vs 79.45 (**+2.8%**, just
above keep-bar; ranges overlap, low confidence).
Prefill: 16k 1613 (+0%), 300k 1586 (+0%), 100k 3× repeats 1645/1653/1609
(≈baseline) → neutral. Ladder dips (8k 1084, 100k 1273, 256k 1350) proven
TRANSIENT (same-boot repeats stable): episodic fabric contention, not
config — screening rule updated: distrust single slow waves, use repeats.
Code inspection: dflash2 speculator shows no draft_parallel usage in the
visible path — TP1 may be partially placebo. Keep (cheap flag, P1 ✓),
final 3-boot verify decides.
Raw: /tmp/expE-prefill.json/.log.

## Exp F — decode stack: EP=1 + DRAFT_TP=1 + --async-scheduling

Rationale: combine the two thin winners (independent mechanisms: target
MoE comm pattern + draft comm) with async scheduling (overlaps CPU
schedule/bookkeeping — incl. spec-decode + 1.3M-pool prefix lookups —
with GPU exec; defaults OFF in this build). If additive ≈ +6% decode.
Gate: P1 + acceptance + C1/C6 + prefill (EP is known −7% prefill; the
stack targets DECODE — prefill must not collapse further).

## Exp F verdict (2026-09-06): SPLIT — C1 +8.3%, C6 −3.2%, prefill −3%

Config: EP=1 + DRAFT_TP=1 + --async-scheduling. All confirmed in log, P1 ✓.
C1 (9 runs): medians 32.08/33.29/29.82 → pooled median **32.08** vs 29.61
(**+8.3%**, above the 5%-real rule — first convincing decode gain).
C6 (6 waves): 75.94/76.04/77.37/76.75/78.25/81.73 → median ~76.9 vs 79.45
(**−3.2%**, 5/6 waves below baseline — real regression).
Prefill: 8k 1504 (−1.7%), 16k 1574 (−2%), 100k 1582 (−4%), 256k 1522
(−5%), 300k 1528 (−4%) → ≈ −3% (milder than EP-alone −7%).
12k rung slow a 4th boot in a row (12.4s) — systematic quirk, ignore.
Hypothesis: async-scheduling helps C1 (overlap) but hurts C6 batching,
or EP×DTP interact. Isolate next.
Raw: /tmp/expF-prefill.json/.log.

## Exp G — EP=1 + DRAFT_TP=1, NO async (isolate async's C1/C6 split)

If C1 stays ≥ +5% and C6 recovers to ≥ −1%: async was C1-friend/C6-foe
(decide then). If C1 drops to ~+3%: async drove the C1 gain (rethink).

## Exp G verdict (2026-09-06): async drove F's C1 gain; EP×DTP poisons C6

Config: EP=1 + DRAFT_TP=1, sync scheduler. P1 ✓.
C1 (9 runs): medians 29.44/30.73/29.82 → pooled ~29.8 vs 29.61 (+0.7%,
gone). F−G ⇒ **async-scheduling contributed ≈ +7.5% C1**.
C6 (4 waves): 77.24/77.53/74.18/77.79 → ~77.4 vs 79.45 (−2.6%, all four
waves low — real). Since F C6 = −3.2% ≈ G C6, async is C6-NEUTRAL; the
C6 poison is EP×DTP interaction (B alone −0.7%, E alone +2.8%, G −2.6%:
non-additive). Mechanism hypothesis: TP1 draft's aux-capture gather vs
EP's sp_all_gather = extra per-step cross-node traffic scaling with
batch; draft was never NCCL-bound, so TP1 adds redistribution tax.
Consequence: DRAFT_TP likely ≤0 everywhere → DROP it. EP-alone (+3% C1
in Exp B) needs re-verification (may be noise; also costs −7% prefill).

## Exp H — async-scheduling ALONE (EP0, no DTP)

Prediction: C1 +7~8%, C6 ≈ baseline. If confirmed, async is a clean
keep and the new stacking base (target: +15% C1).

## Baseline-2 (accidental-but-useful: pure default recipe reboot)

Exp H mis-launched with ASYNC_SCHEDULING=0 (leftover from G) — this boot
is a clean second baseline (EP0, no DTP, sync, eager, align). KEEP IT:
a second baseline boot quantifies boot-to-boot variance, which every
verdict hinges on. Bench C1/C6 + a few prefill rungs, then Exp H
(async-alone) next.

## Baseline-2 verdict: boot variance ±4-7%, drafting is NON-DETERMINISTIC

Pure default recipe, fresh boot. P1 ✓.
C1 (9 runs): medians 34.11/30.87/31.81 → pooled ~31.8 vs baseline-1 29.61
(**+7.4% with ZERO config change**). Fresh-boot C1 range over 7 boots:
29.6–32.1. C6: 78.61/75.49/76.76 → 76.76 vs 79.45 (−3.4%, also boot noise).
Determinism probe: same prompt temp=0 twice → DIFFERENT texts (md5 differ,
len 330 vs 333). DFlash2 drafts probabilistically (gumbel) even at temp 0
→ acceptance/tok/s run-noise ±10% + boot-noise ±5%.
CONSEQUENCES: (a) all ≤5% verdicts (EP +3, DTP +2.8, mamba, graphs) are
UNPROVEN — treat as noise; (b) promote only ≥8% gaps to multi-boot verify;
(c) final verify MUST be 3+ boots/arm (as the goal requires); (d) a true
+15% (≈35.5 C1) clears all observed noise — still provable.
Raw: this boot's C1/C6 above.

## Exp H — async-scheduling ALONE (EP0, no DTP)

Async-boot #2 (F was #1 with EP+DTP). Bar: C1 ≥33 (above fresh-boot range
29.6–32.1) on 9 runs. If it lands ~30 → async is dead, pivot to compile.

## Exp H verdict (2026-09-06): async = CLEAN KEEP (C1 +~5%, C6 neutral)

Config: async-alone (EP0, no DTP). P1 ✓.
C1 (9 runs): medians 31.83/31.25/31.18 → pooled ~31.3. vs B1 29.61 (+5.7%),
vs B2 31.8 (−1.5%). Mid-high of fresh-boot range, 2/2 async boots elevated
(F 32.08, H 31.3). Best estimate +4-6%. C6: 79.50/80.15/80.80 → 80.15
(+0.9% vs B1, +4.4% vs B2) → async is C6-neutral/positive; F's C6 −3.2%
was the EP×DTP poison, not async. KEEP async (no regressions anywhere).
Scoreboard (fresh-boot C1): async ≈ +5%, EP ≈ +3% (1 boot, unverified),
DTP ≈ −2% (DROPPED), graphs 0, mamba 0, MTP impossible.
Combined async+EP (F) ≈ +8%. Need +7% more → torch.compile next.

## Exp I — VLLM_COMPILE (no graphs) on top of async+EP stack

Rationale: only big lever left. Inductor fuses elementwise around marlin
customs + enables comm/compute-overlap passes; helps compute-bound
prefill most, decode latency somewhat. Config: ENFORCE_EAGER dropped,
-cc '{"mode":"VLLM_COMPILE","cudagraph_mode":"NONE"}' (isolate compile,
no P1-risk graphs). Risks: 20-60minрек compile during init, P1 divergence
(gate!), RAM (117GB avail fine). Inductor cache persists on /cache.
Gate: boot + P1 + acceptance + C1/C6 + prefill ladder.

## Exp I config correction (boot-time)

Intended compile-with-NONE-graphs, but `.env CUDAGRAPH_MODE=PIECEWISE`
(leftover) flows into the -cc JSON → actual: **VLLM_COMPILE +
PIECEWISE graphs** + EP + async. Accepted deliberately: graphs-alone
measured 0 (Exp A) so any gain ≈ compile; compile+piecewise is a normal
production combo. Extra gate: P1 must pass (NVFP4 lane once saw P1
divergence with graphs). If P1 ✗ → isolate.
Watch: compile may extend init 20-60min (READY_TIMEOUT=3600; poll manually
past it if needed). Inductor cache persists on /cache.

## Exp I verdict (2026-09-06): FAIL — torch.compile vetoed for Glm5Next

CLI asked VLLM_COMPILE; engine resolved mode=NONE. Cause (vllm/config/
vllm.py): Glm5Next* is on the breakable-cudagraph auto-enable list
("model classes [that] don't carry @support_torch_compile — the breakable
cudagraph is the supported PIECEWISE path") → compile forced NONE.
Overriding (VLLM_USE_BREAKABLE_CUDAGRAPH=0 + decorator patch) = upstream-
uncharted swamp (graph-break storm in mamba/sparse-MLA/marlin, P1 risk).
DEFER unless decomposition says non-MoE dominates. Silver lining: this
boot = EP+async+PIECEWISE (F-redux, 2nd boot) — bench as replication.
P1 gate still required (graphs on).

## Exp I verdict: variance MOB — this boot −10% on identical-ish stack

Config: EP+async+PIECEWISE (no DTP), P1 ✓. C1 medians 31.70/28.99/27.42
(pooled ~28.9, with 26.1/25.5 outlier RUNS) and C6 68.79/72.30/75.00
(median 72.3, −9%) — globally slowest boot yet, same recipe family as F
(+8.3%). Boot-to-boot swing confirmed ±8% BOTH directions. I−F deltas
(+graphs, −DTP) cannot explain −10%; this is a slow boot (or slow hour).
Consequence: single-boot screening is EXHAUSTED — all ≤8% readings are
unresolvable. Path forward: (1) Exp J no-spec decomposition (directs
effort: target-verify vs draft cost); (2) final stack + 3v3 interleaved
verify (goal-mandated); (3) longer runs (1024 tok) to shrink run-noise.
Also: outlier RUNS (25-26 tok/s) within boots ⇒ acceptance-luck run-noise
is large; use 1024-token runs henceforth.
Raw: this turn's C1/C6 above.

## Exp J — SPEC_METHOD=none (pure target, no drafter)

Decomposition boot. Measures: (a) target-only step cost (C1 tok/s =
steps/s, 1 tok/step); (b) target-only prefill (draft-prefill fraction =
spec-prefill vs J-prefill); (c) whether spec overhead (draft+verify-M8
vs plain-M1) is where time goes. Gate: boot + P1 (no acceptance).
Then: micro-stack decision + final 3v3.

## Exp J verdict: DECOMPOSITION (no-spec boot, P1 ✓)

Pure target, eager/EP0/sync: C1-1024 ×4 → **14.15–14.23 (±0.3%, rock
stable, deterministic)** — engine has NO jitter; spec run-noise is 100%
acceptance luck (diffusion seeds). Per-token: no-spec 70ms vs spec
33ms (2.1× — same ratio class as Tony's MTP-4→DFlash 2.15×).
Prefill (no-spec): 16k 1604, 100k 1638/1641 — IDENTICAL to spec boots ⇒
draft-prefill ≈ 0 (vLLM skips draft work on pure-prefill steps). Prefill
= pure target-MoE-bound, spec-INDEPENDENT. EP's −7% prefill was target-
comm (allreduce vs dispatch at M=8192).
Step model (spec C1): ~140ms = verify-M8 (~80-100?) + draft (~40-60?) +
sched. Draft fraction ~30-40% ⇒ 2× faster draft ≈ +17% decode. Draft-KV
dtype is the first draft knob (bf16 skips quant kernels).

## Exp K — draft kv_cache_dtype=auto(bf16) + EP + async (stack)

Config: dflash K7 + spec-JSON `"kv_cache_dtype":"auto"` (DRAFT bf16 KV;
target stays fp8_e4m3) + EP1 + ASYNC1, eager, align. Same trick the SM120
overlay uses for compat — here for speed (skip fp8 quant/dequant on the
latency-sensitive draft path; draft ctx small so 2× bytes is trivial).
Doubles as async+EP-stack replication (3rd async boot).
Gate: boot + P1 + acceptance (MUST stay ~0.42 — dtype must not change
draft numerics... it shouldn't, KV precision only) + C1-1024×5 + C6×3.

## Exp K verdict (2026-09-06): FAIL — draft bf16 KV kills acceptance

Config: spec `"kv_cache_dtype":"auto"` (draft bf16) + EP + async. Boot OK,
P1 ✓. Acceptance (large sample, 9331 drafts): **0.332 vs 0.41-0.42
baseline (−20%)** — draft numerics are finely tuned for fp8 KV; bf16
drafts agree less with the fp8-KV target?! (Or SWA-bf16 path differs.)
C1-1024 median 25.8 (consistent: −20% accepts/step ≈ −20% tok/s).
REVERT to inherited fp8 draft KV. Lesson: acceptance is FRAGILE — do not
touch draft numerics.
Also learned: 1024-runs need their own baseline (ctx effects) — keep
512-runs as the comparable screening metric + add 1024 at verify.
Raw: this turn.

## Exp L1 — async ALONE + micro-bundle (EP0, draft fp8)

Micros (env passthrough): VLLM_MARLIN_USE_ATOMIC_ADD=1 (small-n marlin
linears), VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 (router fusion). Both ~0-1%
hoped, P1-gated. EP0 (isolate; EP re-added in L2 if L1 good). This is
also the async-prefill measurement boot (H never measured prefill).

## Clocks discovery (2026-09-06): GPU at 2281MHz, max 3003!

`nvidia-smi -q -d CLOCK`: runs 2281, app-clock 2418, max 3003. No sudo —
but docker-group root-equivalent works: `docker run --rm --privileged
--gpus all <image> nvidia-smi -lgc 3003,3003` on BOTH nodes → settles
~2509/2489MHz (+10%, power 28→44W, SM 95→89%: bottleneck shifts toward
UMA/fabric). Legit recipe step (host tuning, same model/quant, P1-safe,
re-apply after GPU reset; will wire into start.sh + docs).
L1-locked (async+micros, EP0): C1 pooled ~32.5 (+7% from clocks);
C6 median ~78.5 (~0%); prefill 8k 1602/16k 1652/100k 1687/256k 1632
(+2-4% — prefill is UMA-bandwidth-bound, clocks barely help).
Prefill +15% looks UNREACHABLE (all knobs exhausted, physics-bound).
Decode path: L1-locked +9.8% vs B1. Need ~+5% more → FULL graphs (Exp M).

## Exp M — FULL cudagraphs + async + EP + micros (locked clocks)

Last untested big decode lever here (PIECEWISE was neutral; NVFP4 lane
FULL +4.3% once with P1 divergence → STRICT gate). Config: ENFORCE_EAGER=0,
-cc '{"mode":"NONE"? no — mode stays default (NONE only if vetoed...)}'
actual: EAGER dropped, -cc cudagraph_mode=FULL (breakable auto for
Glm5Next; may downgrade w/ warning per #49547 — read logs). EP1 ASYNC1
micros on. Clocks re-locked post-boot (check persistence first).
Gate: boot + P1 STRICT + acceptance ~0.42 + longer-output eyeball +
C1-512×5 + C6×3. Any divergence → revert to eager, FULL is dead.

## Exp M verdict: STEP RATE +18%, tok/s +11% (acceptance luck masks it)

Config: FULL_DECODE_ONLY graphs (FULL captured target+drafter, 6/6 each)
+ EP + async + micros + locked clocks (2587/2541MHz). P1-STRICT ✓ (3/3),
acceptance 0.451 healthy.
C1 (12 runs): median **32.9** vs B1 29.61 (+11.1%), vs B2 31.8 (+3.5%).
C6: 76.91/85.48/81.42 → 81.42 (+2.5%).
KEY DIAGNOSTIC (6 runs w/ per-run accept deltas): tok/s ÷ accepts/step =
CONSTANT 8.2-8.3 steps/s ⇒ step time CONSTANT, 100% of run-variance is
acceptance luck (meanlen 3.64–4.23). Step rate: M 8.25/s vs B1-era ~6.9/s
≈ **+18-20%**. Tok/s medians understate it (luck). Verify with VOLUME
(9+ runs/boot) will reveal it. Drafter = single 5-layer forward (~10-15%
of step); target-verify dominates; all flag levers now applied.
No further big levers: prefill physics-bound (~+3% clocks), acceptance
fragile/fixed, compile vetoed, MTP impossible.
DECISION: freeze M-stack as FINAL, wire clocks into recipe, 3v3 verify.

## FINAL VERIFY plan (goal-mandated 3×3, interleaved)

Arms: BASE = `.env.base` (default recipe) + stock clocks (`clocks.sh reset`
post-boot); FINAL = `.env` (M-stack: FULL graphs+EP+async+micros) + locked
clocks (start.sh hook). Order B1,F1,B2,F2,B3,F3. Per boot: verify_final.sh
(P1 gate + acceptance + 9× C1-512 + 3× C6-512 + 2× 100k-prefill).
Bar: FINAL median ≥ BASE median × 1.15 on C1-decode (goal). C6/prefill
reported honestly (expected: C6 small+, prefill ±few %).
Rationale for BASE-unlocked vs FINAL-locked: clock-lock is a kept recipe
step (host tuning, same model/quant, P1-safe); the comparison is
as-found-server vs improved-recipe.

## VERIFY B1 (baseline, stock clocks, fresh boot): C1 30.78 / C6 79.16 / 100k 1664

9× C1-512: 32.74/34.25/28.52/35.40/29.50/30.78/35.35/29.10/28.00 →
median **30.78** (fresh baselines run higher than the 23h-stale 29.61).
C6: 79.16/78.80/82.00 → **79.16**. 100k×2: 1656/1672 → **1664**.
P1 ✓. Raw: benchmarks/verify/B1/.

## VERIFY F1 (FINAL stack, locked): C1 33.03 (+7.3%) / C6 76.8 (−3%) / 100k 1570 (−6%)

9× C1: median **33.03** vs B1 30.78 (+7.3%). C6: 79.28/76.80/75.23 →
76.80 (−3.0%). 100k×2: 1539/1602 → 1570 (−5.6%, EP poison confirmed).
P1 ✓, acceptance 0.535 (lucky). Worker clocks failed to lock in hook
(eval quoting — fixed after; re-locked manually, both ~2560).
ASSESSMENT: C1 +7% is real-ish but HALF of +15%; C6/prefill regress via
EP. Verify of this stack would fail the goal. PAUSE verify (B1/F1 kept
as variance points). Last big lever: force torch.compile past the
breakable veto (Exp N). If N fails → fall back to no-EP FINAL (~+5%,
honest non-regression) and reassess (likely: block-size-1152 prefix play
or admit ~+10% max and keep grinding new ideas).

## Exp N — force VLLM_COMPILE past breakable veto (EP0, graphs NONE, async, micros, locked)

Code reading: the veto lives in vllm/config/vllm.py (Glm5Next* → breakable
auto → mode=NONE). The @support_torch_compile decorator only ADDS
do_not_compile flags; an UNDECORATED model has no such attr, so with
VLLM_USE_BREAKABLE_CUDAGRAPH=0 + mode=VLLM_COMPILE, dynamo WILL trace it
(breaks where it must). Untested upstream = risk, not certainty.
Config: VLLM_USE_BREAKABLE_CUDAGRAPH=0, -cc '{"mode":"VLLM_COMPILE",
"cudagraph_mode":"NONE"}', EP0, ASYNC1, micros on, eager dropped by logic.
Gate: (a) init duration (7min = vetoed/skip; 20-60min = compiling);
(b) boot success (opaque custom ops are inductor-safe; mamba/indexer may
break or fail); (c) STRICT P1 + acceptance; (d) C1/C6/prefill.

## Exp N verdict: compile ≈ graphs (both +0-1% over clocks+async+micros)

VLLM_COMPILE forced past veto (VLLM_USE_BREAKABLE_CUDAGRAPH=0): dynamo
traced, AOT artifacts saved, boot normal length, P1-STRICT 3/3 ✓,
acceptance 0.430 ✓. C1-512×9 median **32.61** vs M-stack 32.9 (tie) vs
B1 30.78 (+6%). So compile-no-graphs ≈ graphs-no-compile ⇒ fusing
elementwise already ~0% (step is GEMM/memory-bound customs). Keep GRAPHS
(faster boot, P1-proven 2 boots), drop compile.
Honest ledger: flags ≈ +2-4%, clocks ≈ +7%, graphs/compile ≈ +0-1%.
TOTAL ≈ +10-12%. +15% needs more — next: draft-FP8 + LL128 (Exp O).

## Exp O — draft quantization=fp8 + NCCL_PROTO=LL128 (EP0 stack)

(a) Draft is bf16 dense 1B (~15ms/step?). FP8 drafter halves its weight
reads (2.2→1.1GB) — hoped +2-3% decode IF acceptance holds (fragile!).
spec-JSON "quantization":"fp8". Gate: boot + P1 + acceptance≈0.42.
(b) LL128: lower latency for small decode allreduces (+1% hoped);
prefill-bandwidth risk — measured, dropped if prefill regresses.
Base: FULL graphs + async + micros + locked clocks, EP0 (honest product:
no C6/prefill regressions allowed in FINAL).

## Exp O verdict (partial): draft-fp8 BOOT-FAILS — abandoned

`quantization:fp8` in spec JSON → draft loader dies:
`ValueError: hf_overrides must be a dict for get_quant_config`. The draft-
quant plumbing isn't wired for the dflash path in this build (eagle/mtp
only?). Not worth spelunking for a hoped +2-3%. DRAFT_QUANT reverted.
Continuing O as LL128-only test (same stack otherwise).

## Exp O verdict: LL128 −2.8% (revert to NCCL auto), draft-fp8 unbootable

O2 (LL128 + FULL + async + micros, EP0, locked): C1 medians
30.43/31.62/31.79 → ~31.6 vs L1-locked 32.5 (−2.8%). LL128 hurts
(small-decode NCCL was already auto-optimal). REVERT to auto.
Draft-fp8 unbootable (dflash path plumbing broken). Both dropped.
Best cold stack stands: M (EP1, +11% C1 but −6% prefill) / L1-locked
(EP0, +10% C1, prefill +3%). Honest FINAL leans L1-locked (no regressions).

## Exp P — BLOCK_SIZE=1152 (prefix-threshold play + speed check)

Rationale: prefix hits need ≥2 full blocks (4609 tok @2304). @1152 the
threshold may drop to ~2305 → 2.5–4.5k-token agent turns go from 0% to
~85% hits = +90% EFFECTIVE prefill on agentic workload (custom benchmark
sanctioned by goal). Cold ladder/decode must NOT regress (else drop).
Risks: mamba-page math (runtime doubles attn block to 4608 — 1152 may
fail slot-share divisibility → boot gate decides); mamba-align group may
floor the threshold regardless; 2× blocks = trivial metadata+.
Gate: boot + KV pool + P1 + prefix Ladder (3.6k/7.2k prompts) + C1/C6.

## Exp P verdict: FAIL — 1152 kills 5k-turn prefix hits, revert to 2304

Block 1152 boots fine (pool intact, P1 ✓) but prefix: 3.6k→0 hits,
5k→0 hits (was 4608 @2304!), 7.2k→4608 hits. Hit quantum is the 4608
EFFECTIVE attention block (mamba floor), not --block-size; smaller blocks
only fragment (more eagle-drop loss). 2304 pin RESTORED. (Larger blocks
untestable for same reason — 4608 gave 0 hits: breaks mamba-align.)

## Exp Q verdict: KEEP (+2.8% C1, +3% C6, prefill neutral, P1 ✓)

`--language-model-only` (EP0, FULL, async, micros, locked): vision tower
+ mm_prefix attention mode skipped ("running in text-only mode").
C1 (9 runs): medians 33.39/31.94/33.71 → pooled **~33.4**
(+12.8% vs B1 29.61, +8.5% vs B1-verify 30.78).
C6: 82.60/78.96/81.74 → **81.74** (+2.9%/+3.3%).
Prefill: 16k 1569 (−3%), 100k 1682/1698 (+2-3%) → neutral.
Acceptance 0.426 (24k drafts ✓). Model −0.5GiB (83.58 vs 84.12).
Mechanism: per-step mm_prefix attention-mode overhead removed (backends
unchanged: MLA-sparse-SM90 + FI/xqa). TRADEOFF: vision serving disabled
(text-only; document as a text-serving variant).
Micros note: fused-grouped-topk needs e_score_correction_bias (absent
here?) and atomic-add needs n<2048 (MoE n≥4096) → both likely INACTIVE;
kept (harmless, future shapes).

## Exp R — Q + EP (does EP stack on lm-only? + prefill cost?)

If C1 ≥33.7 with prefill ≥1650 → FINAL=R. If prefill tanks (<1600) or
C1 flat → FINAL=Q (EP0, no regressions).

## Exp R verdict: EP is POISON — dropped permanently (FINAL=Q-stack EP0)

Q+EP (locked): C1 medians 31.17/31.14/31.49 (~31.3, −6% vs Q 33.4),
C6 76.9 (−6%). Global slowdown (or slow boot + EP poison). EP rap sheet:
B +3% (1 boot, noise), F1-prefill −6%, G-C6 −2.6%, R −6% everywhere.
VERDICT: EP never helps reliably; hurts prefill/C6 consistently. DROPPED.
FINAL (frozen): locked clocks + FULL graphs + async + micros + lm-only +
dflash K7 + 2304 + EP0 + NCCL auto. Best: C1 33.4 / C6 81.7 / 100k ~1690.
Micros confirmed ACTIVE (noaux_tc bias→fused topk; gate n=288→atomic-add).
top_k=1 harness tweak = 0 (warmup artifact). EPLB off by default (n/a).
Drafting nondeterminism (diffusion seeds) confirmed as run-noise source;
engine itself stable (no-spec ±0.3%).

## FINAL 3v3 VERIFY (6 fresh boots, interleaved B/F)

BASE=.env.base+stock clocks vs FINAL=.env+locked. verify_final.sh each
(P1+accept+9×C1+3×C6+2×100k). Bar: FINAL ≥ BASE×1.15 on C1-decode.
Prior points (stale/mixed configs) retired to screening history.

## Mixed-decode metric (decode-under-load): BASELINE vB1 = 12.206

bench_mixed_prefill (1024 decode + ~7000-repeat cold prefill, partial
overlap): vB1 runs 12.206/12.204/12.488 → median **12.206** (tight ±1%:
1024 tokens average acceptance luck; chunk schedule deterministic).
FINAL-needs: ≥14.04 (+15%). Mechanism for compounding: faster chunks
(clocks+async+FULL) unblock decode sooner AND faster steps — could exceed
pure-C1 gains. Old 8192-vs-16384 A/B reference: 11.91 (same family ✓).
vB1 full: C1 31.56 / C6 78.72 / 100k 1648 / mixed 12.206. Raw: verify/vB1/.

## VERIFY vF1 (FINAL): C1 +4.6% / C6 +5.0% / prefill +1.8% / MIXED −7.8%

FINAL (Q-stack EP0, locked): C1 median 32.18 (28.8–36.1) vs vB1 31.56
(+4.6%); C6 80.51/83.37/82.68 → 82.68 (+5.0%); 100k 1671/1685 → 1678
(+1.8%); MIXED 11.69/10.69/11.26 → 11.26 vs vB1 12.21 (**−7.8%**).
Sobering: screening overestimated (stale B1 + lucky boots); TRUE flag
effect ≈ +5%. Mixed regression suspects: FULL graphs (shape-varying
chunk+decode batches → graph misses/fallbacks under load) and/or async
batching. A serving recipe can't regress loaded serving → must fix.
Raw: verify/vF1/ (+ screening vB1).

## Exp T — mixed-friendly: 4096-chunks + EAGER + SYNC (drop graphs/async)

Theory: decode-under-load ∝ chunk blocking time (16384 hurt −8% ⇒ 4096
should help symmetrically +~8%); graphs/async suspected load-fangled
(shape-varying batches miss graphs; async batches worse under overlap).
T = EAGER + SYNC + EP0 + lm-only + micros + 4096 + locked (keep clocks,
lm, micros — proven load-neutral/positive).
PRIMARY: mixed×3 (needs ≥14.04 = vB1-mixed 12.21 × 1.15).
GATES (no regressions): pure C1/C6 ≥ vB1, pure prefill ≥ ~1640.
If T-mixed wins with pure intact → new FINAL, restart 3v3 around T.

## Exp U — NCCL_PROTO=LL + NCCL_ALGO=Tree (lowest-latency path)

Last cheap knob (LL128 hurt; plain LL + Tree untested). Hoped +1-2%
decode (150 small allreduces/step). Prefill-bandwidth risk → gate pure
prefill too. Stack: Q-EP0 + 8192 (reverted T's 4096) + U.
Math check: 1.07(clock)×1.025(async)×1.028(lm)×1.005(graphs)×1.003(micros)
≈ +13.5% → +NCCL(1%) ≈ +14.6% → 3v3 could land ≥15% with alignment.

## Exp U verdict: NCCL tuning DEAD (auto optimal)

Tree+LL → boot HARD FAIL: `NCCL WARN No algorithm/protocol available for
AllGather with ncclInt8` (Tree lacks LL support for some collectives on
this RoCE fabric; no fallback). LL128 hurt (−2.8%). NCCL auto stands.
Reverted to auto. FINAL frozen (Q-stack EP0 8192): locked + FULL + async
+ micros + lm-only.

## Fabric health: PRISTINE (transients are engine-side)

rocep1s0f0: xmit_discards=0, rcv_errors=0, symbol_error=0, ECN-marked=0,
CNP=0, out_of_buffer=0, transport/rnr retries=0, xmit_wait=0. No loss, no
congestion. Transients (12k-curse, ladder dips) are engine-side (not
fabric); medians aren't fabric-dragged. Fabric tuning closed for good.
NCCL auto optimal (LL128 hurts, Tree+LL incompatible).

## VERIFY status (interleaved 3v3, FINAL=Q-stack EP0 locked)

- vB1 BASE (stock, unlocked both): C1 31.56 / C6 78.72 / 100k 1648 / mixed 12.21.
- vF1 FINAL (locked/FULL/async/micros/lm-only): C1 32.18 / C6 82.68 / 100k 1678.
  Gaps: C1 +2.0% / C6 +5.0% / pre +1.8%. (vB1 baseline running strong;
  screening's +12% was stale-B1 + lucky-boot inflated. TRUE ≈ +5%.)
- In flight: vF2 (FINAL). Remaining: vB2, vB3, vF3.
Raw: benchmarks/verify/vB1, verify/F1(retired M-config), verify/vF1/.

## Exp V — scheduling-policy=priority (mixed-decode lever)

Scheduler flags found: --scheduling-policy (fcfs/priority),
--prefill-schedule-interval, --max-num-scheduled-tokens.
Theory: priority lets decode preempt/prioritize over prefill chunks →
mixed-decode UP with pure metrics unaffected (no contender when solo).
PRIMARY: mixed×3 (need ≥14.04 = +15% over vB1-mixed 12.21).
GATES: pure C1/C6/prefill must hold (priority is no-op solo).
Config: Q-stack + --scheduling-policy priority (+ prefill-interval default).

## Uptime-decay hypothesis (test in 3v3 going forward)

Fresh baselines creep up (B1-23h 29.61 → B2 31.8 → vB1 31.56 → vB2 32.64)
while the stack sits 30-33. Either fresh boots are fast (caches warm) or
23h-uptime decays (KV fragmentation over hundreds of bench requests).
If decay is real, fair A/B needs time-matched boots (interleaved 3v3 does
this ✓). Protocol add: per-boot C1-early (post-warmup) + C1-late (after
suite, ~40min apart) to measure slope. Drafter confirmed single-forward
(no iterations to cut); Humming/CUTLASS/TRTLLM all need FP4/FP8 (INT4
locked out) — the quant IS the ceiling (NVFP4 lane rides fast paths).
Marlin/FA2-compat + fixed acceptance + capped clocks + UMA = the wall.

## VERIFY vB3 BASE: C1-early 30.91 / C6 76.65 / 100k 1651 / C1-late 30.17

Early (9 runs) 30.91 vs late (5 runs, ~50min later) 30.17: −2.4%, mild
uptime-decay signal (consistent with stale-B1 depression; interleaving
covers it). Fresh-BASE mean ≈ 31.5 (B2 31.8, B1v 30.78, vB1 31.56, vB2
32.64, vB3 30.91). FINAL mean ≈ 31.9 (Q 33.4, vF1 32.18, vF2 30.18).
TRUE GAP ≈ +1-2% — screening estimates were luck-inflated (1-boot arms).
Only clean within-boot evidence stands: clocks +5% (L1). The +15% needs
a heroic vector (FI rebuild?); 3v3 completes honestly regardless.
Raw: benchmarks/verify/vB3-early... (this turn; full suite logged).

## FECHAMENTO (limpeza solicitada): stack final = clocks 2400 + async + micros

Decisões: clocks em 2400 (não 2500 — calor), SEM language-model-only
(vision importa no dia a dia), COM async, SEM cudagraphs (marginal).
Removidos do `.env`/launcher: EP, DRAFT_TP/KV/QUANT, COMPILE/CUDAGRAPH,
NCCL_PROTO/ALGO, MAMBA, MM, SCHED_POLICY. Mantidos: async, micros,
ENV_FILE (A/B), hook de clocks + `clocks.sh reset`. `.env.base`
regenerado limpo; `.env.final` e B1/F1 antigos removidos; README ganhou
seção "Performance tuning".
Confirmação pós-restart (eager, vision on, 2400MHz): P1 ✓, vision "Red" ✓,
C1 31.24 (29-34), acceptance 0.402, C6 79.0. Sob C6: 64-67°C, 33-35W,
clocks segurando 2398 — longe de throttle. +15% single-metric segue
inatingido (teto estrutural documentado acima); ganho honesto banked ~+8%.
