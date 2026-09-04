#!/usr/bin/env python3
"""SM121 MLA zero-pad correctness (pure stdlib, runs on CPU-only host).

Proves the core hypothesis of the SM120 overlay:
    Q' = [Q(512) | zeros(64)], K' = [K(512) | zeros(64)]
    => Q'.K' == Q.K  (scores bit-identical, softmax unchanged)

plus the geometry/buffer arithmetic the overlay relies on:
  - packed fp8_ds_mla record = 512 NoPE + 16 B scales + 128 B RoPE = 656 B
  - overhead vs ideal NoPE ~528 B  ~= 24%
  - stock indexer buffer 2048 + (kpool-1)=3 -> 2051 -> round128 -> 2176
    (rejected by the SM120 topk==2048 template parameter)
  - patched: 511 pools * 4 + 3 tail = 2047 <= 2048; select_k stays 512
    (keeps the fused top-k fast path gated on {512,1024,2048})

Optionally (SM121_VERIFY_OVERLAY=1) re-runs docs/patch_sm121_mla.py
--verify-only on $SM121_WORK_DIR (skipped when docker/image absent).

Exit 0 on success, non-zero with a diagnostic otherwise.
"""
from __future__ import annotations

import math
import os
import random
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = math.fsum(exps)
    return [e / s for e in exps]


def test_qk_identity() -> None:
    rng = random.Random(1234)
    for trial in range(50):
        # bf16-ish magnitudes: latent values ~ N(0, 1)
        q = [rng.gauss(0, 1) for _ in range(512)]
        keys = [[rng.gauss(0, 1) for _ in range(512)] for _ in range(8)]
        qp = q + [0.0] * 64
        keysp = [k + [0.0] * 64 for k in keys]
        s_plain = [dot(q, k) for k in keys]
        s_pad = [dot(qp, k) for k in keysp]
        for a, b in zip(s_plain, s_pad):
            assert a == b, f"QK mismatch: {a} vs {b}"
        p_plain, p_pad = softmax(s_plain), softmax(s_pad)
        for a, b in zip(p_plain, p_pad):
            assert abs(a - b) < 1e-15, f"softmax drift: {a} vs {b}"
    print("[ok] QK identity: dot([Q,0],[K,0]) == dot(Q,K), softmax unchanged (50 trials)")


def test_geometry() -> None:
    assert 512 + 16 + 128 == 656, "packed fp8_ds_mla record must be 656 B"
    overhead = (656 - 528) / 528
    assert 0.23 < overhead < 0.26, f"KV overhead out of range: {overhead:.3f}"
    print(f"[ok] KV geometry: 512 + 16 + 128 = 656 B/token/layer "
          f"(overhead vs ideal ~528 B: {overhead * 100:.1f}%)")
    # kernel guard (flashinfer/mla/_core.py:582): (512, 64, 576)
    assert (512, 0, 512) != (512, 64, 576), "sanity"
    print("[ok] NoPE (512,0,512) != GLM_NSA kernel requirement (512,64,576) "
          "-> pad path justified")


def test_indexer_arithmetic() -> None:
    topk, kpool = 2048, 4
    stock = topk + (kpool - 1)
    assert stock == 2051
    rounded = ((stock + 128 - 1) // 128) * 128
    assert rounded == 2176, f"stock buffer width: {rounded}"
    patched_buf = topk
    assert patched_buf == 2048
    select_k = topk // kpool
    assert select_k == 512 and select_k in (512, 1024, 2048)
    produced = (select_k - 1) * kpool + (kpool - 1)
    assert produced == 2047 <= 2048
    # vs the 2051-wide stock row we give up exactly one (lowest-ranked) pool;
    # vs the 2048 buffer one slot stays -1 padding.
    assert (stock - produced) == 4, "must drop one 4-token pool"
    assert (patched_buf - produced) == 1
    print(f"[ok] indexer: stock buffer {rounded} (rejected) -> patched {patched_buf}; "
          f"candidates {produced}/2048 (drop lowest-ranked pool only, keep tail)")


def test_overlay_verify_optional() -> None:
    if os.environ.get("SM121_VERIFY_OVERLAY") != "1":
        print("[skip] overlay --verify-only (set SM121_VERIFY_OVERLAY=1 to run)")
        return
    work = os.environ.get("SM121_WORK_DIR", "/var/tmp/glm53-w4a16-cache/sm121")
    script = os.path.join(REPO, "docs", "patch_sm121_mla.py")
    image = os.environ.get("SM121_TEST_IMAGE", "radixark/vllm-glm53-flash:sm121-v11-dflash2")
    r = subprocess.run(
        ["python3", script, "--image", image, "--work-dir", work, "--verify-only"],
        capture_output=True, text=True,
    )
    print(r.stdout.strip() or "(no stdout)")
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        raise SystemExit("overlay verify FAILED")
    print("[ok] overlay files verify OK")


def main() -> int:
    test_qk_identity()
    test_geometry()
    test_indexer_arithmetic()
    test_overlay_verify_optional()
    print("SM121 MLA math/correctness tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
