#!/usr/bin/env python3
"""GLM-5.3-Flash NoPE sparse MLA on SM121 — feature-gated overlay generator.

Generates bind-mountable patched vLLM files that enable
``FLASHINFER_MLA_SPARSE_SM120`` for the rope-free GLM-5.3 MLA
(``qk_rope_head_dim=0``, ``kv_lora_rank=512``) by reusing the 576-wide
GLM_NSA kernel geometry via zero padding. Ported semantically from the
functional reference ``MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks``
Dockerfile section "GLM-5.3-Flash NoPE sparse MLA on SM121" to our pinned
tree (vLLM ``0.1.dev20051+g487ecf187`` — same commit as the reference base,
all anchors verified byte-exact).

Math (exact, no approximation):
    Q' = [Q(512) | zeros(64)],  K' = [K(512) | zeros(64)]
    Q'.K' = Q.K + 0  -> attention scores bit-identical; value comes from the
    512-dim NoPE region (d_v=512), so the output is exact. Cost: 656 B/token
    instead of ~528 (~24% more DSA KV).

What is patched (MLA/KV subset ONLY — no EXL3/MoE/Trellis, no warmup/PDL
neutering, no weight changes; Marlin W4A16 untouched):
  1. v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py
     - supports_dense_mha_prefill = False (prefill uses sparse top-k MQA path)
     - rope_pad = 64 when qk_rope_head_dim == 0 (+ strict geometry asserts)
     - Q 512 -> 576 zero pad at kernel boundary; kernel gets qk_rope_head_dim=64
     - return_valid_counts + per-row topk_lengths as seq_lens; capacity as
       max_seq_len / sparse_mla_top_k; empty-row masking
     - do_kv_cache_update override: synthetic k_pe zeros(..., 64) so
       concat_and_cache_mla writes the packed fp8_ds_mla 656 B record
  2. models/glm5next/nvidia/model.py + mtp.py
     - buffer_width = topk_tokens (drop the +kpool-1 tail widening + 128-round
       that produced a 2176-wide table the SM120 template rejects)
  3. v1/attention/backends/mla/flashinfer_mla_sparse.py
     - SM120 kernel block sizes [64, 256] -> [64] (GLM_NSA/DSV3_2 kernels are
       instantiated at PAGE_BLOCK_SIZE=64 only)
  4. platforms/cuda.py
     - indexer block alignment uses page 64 on major==12 (DeepGEMM sm_120
       only accepts block_kv=64 for non-FP4 cache)
  5. model_executor/layers/sparse_attn_indexer_kpool.py (prefill + decode)
     - expand pool_ids[:, :select_k-1]: drop the lowest-ranked pool instead of
       the recent tail -> 511*4 + 3 tail = 2047 candidates in a 2048 buffer
       (keeps the fused select_k=512 fast path; gives up 0.2% lowest-ranked)

Deliberately NOT ported (test-first per bring-up plan):
  - FlashInfer sparse-MLA warmup/autotune skip (our boot already passes
    --no-enable-flashinfer-autotune; fused_moe autotune shows no hang here)
  - PDL gate change (our tree already has `major in (9, 10)`)

Fail-closed: every anchor must match exactly once or the script aborts with
no output written. Idempotent: re-running on already-patched files is a no-op
(markers detected). The existing SM121 persistent_topk indexer fix
($PATCH_KPOOL_HOST / docs/sparse_attn_indexer_kpool_sm121.py) is preserved by
using it as the indexer base when present, so flag=1 is a strict superset of
flag=0 behaviour.

Usage:
  python3 docs/patch_sm121_mla.py --image IMG --work-dir DIR \
      [--base-indexer PATH] [--verify-only]

Markers: [glm53-sm121-mla] (port), [SM121 MLA] (our gates/logs).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MARK = "[glm53-sm121-mla]"
SITE = "usr/local/lib/python3.12/dist-packages/vllm"

FILES = {
    "sm120.py": "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    "sparse.py": "v1/attention/backends/mla/flashinfer_mla_sparse.py",
    "cuda.py": "platforms/cuda.py",
    "indexer.py": "model_executor/layers/sparse_attn_indexer_kpool.py",
    "glm_model.py": "models/glm5next/nvidia/model.py",
    "glm_mtp.py": "models/glm5next/nvidia/mtp.py",
}

OUT_NAMES = {
    # work-dir filename -> container absolute path
    "flashinfer_mla_sparse_sm120.py": f"/{SITE}/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
    "flashinfer_mla_sparse.py": f"/{SITE}/v1/attention/backends/mla/flashinfer_mla_sparse.py",
    "cuda.py": f"/{SITE}/platforms/cuda.py",
    "sparse_attn_indexer_kpool.py": f"/{SITE}/model_executor/layers/sparse_attn_indexer_kpool.py",
    "glm5next_model.py": f"/{SITE}/models/glm5next/nvidia/model.py",
    "glm5next_mtp.py": f"/{SITE}/models/glm5next/nvidia/mtp.py",
}


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"anchor error [{label}]: expected 1 match, found {n}")
    return text.replace(old, new, 1)


def extract_pristine(image: str, dest: Path) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    cid = subprocess.run(
        ["docker", "create", image], capture_output=True, text=True, check=True
    ).stdout.strip()
    try:
        out: dict[str, str] = {}
        for key, rel in FILES.items():
            p = dest / f"pristine_{key}"
            if not p.is_file():
                subprocess.run(
                    ["docker", "cp", f"{cid}:/{SITE}/{rel}", str(p)],
                    check=True,
                    capture_output=True,
                )
            out[key] = p.read_text()
        return out
    finally:
        subprocess.run(["docker", "rm", cid], capture_output=True, check=False)


def patch_sm120(text: str) -> str:
    if MARK in text:
        return text
    # R0: logger import + module logger (our addition for §9 logs).
    text = replace_once(
        text,
        "from vllm.v1.attention.backends.mla.sparse_utils import (\n"
        "    triton_convert_req_index_to_global_index,\n"
        ")\n",
        "from vllm.v1.attention.backends.mla.sparse_utils import (\n"
        "    triton_convert_req_index_to_global_index,\n"
        ")\n"
        "from vllm.logger import init_logger\n",
        "sm120-logger-import",
    )
    text = replace_once(
        text,
        "if TYPE_CHECKING:\n"
        "    from vllm.model_executor.models.deepseek_v2 import Indexer\n",
        "if TYPE_CHECKING:\n"
        "    from vllm.model_executor.models.deepseek_v2 import Indexer\n"
        "\n"
        "\n"
        f'logger = init_logger(__name__)  # {MARK} §9 sanity logs\n',
        "sm120-logger",
    )
    # R1: supports_dense_mha_prefill = False (reference).
    text = replace_once(
        text,
        '    """SM120 FlashInfer sparse-MLA implementation."""\n\n    is_sparse = True\n',
        '    """SM120 FlashInfer sparse-MLA implementation."""\n\n'
        "    is_sparse = True\n"
        f"    supports_dense_mha_prefill = False  # {MARK} prefill via top-k MQA\n",
        "sm120-prefill-flag",
    )
    # R2: rope_pad + kernel geometry + strict gates (reference + §9 asserts).
    text = replace_once(
        text,
        '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
        "        from vllm.config import get_current_vllm_config\n",
        '        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]\n'
        "        from vllm.config import get_current_vllm_config\n"
        f"        self.rope_pad = 0  # {MARK} NoPE->GLM_NSA compat\n"
        "        if self.qk_rope_head_dim == 0:\n"
        "            if self.kv_lora_rank != 512:\n"
        "                raise NotImplementedError(\n"
        '                    "FLASHINFER_MLA_SPARSE_SM120 pads NoPE MLA into the "\n'
        '                    "576-wide GLM_NSA geometry, which requires "\n'
        '                    f"kv_lora_rank=512; got {self.kv_lora_rank}."\n'
        "                )\n"
        "            self.rope_pad = 64\n"
        "        self.kernel_qk_rope_head_dim = self.qk_rope_head_dim + self.rope_pad\n"
        f"        # {MARK} sanity gates: known-good GLM-5.3 geometry only.\n"
        "        if self.rope_pad:\n"
        "            if self.kernel_qk_rope_head_dim != 64:\n"
        "                raise RuntimeError(\n"
        '                    "[SM121 MLA] kernel_qk_rope_head_dim must be 64 "\n'
        '                    "when rope_pad is active; got "\n'
        "                    f\"{self.kernel_qk_rope_head_dim}.\"\n"
        "                )\n"
        "            _hf = getattr(\n"
        '                get_current_vllm_config().model_config, "hf_text_config", None\n'
        "            )\n"
        '            _index_topk = getattr(_hf, "index_topk", None)\n'
        "            if _index_topk is None or int(_index_topk) != 2048:\n"
        "                raise RuntimeError(\n"
        '                    "[SM121 MLA] rope_pad path requires index_topk=2048; "\n'
        '                    f"got {_index_topk}."\n'
        "                )\n"
        "            try:\n"
        "                _major, _ = torch.cuda.get_device_capability()\n"
        "            except Exception:\n"
        "                _major = None\n"
        "            if _major is not None and _major != 12:\n"
        "                raise RuntimeError(\n"
        '                    "[SM121 MLA] rope_pad path requires SM12x "\n'
        '                    f"(major==12); got major={_major}."\n'
        "                )\n"
        "            logger.info_once(\n"
        '                "[SM121 MLA] GLM-5.3 NoPE compatibility mode enabled: "\n'
        '                "Q 512 -> 576 zero padded, K_PE 0 -> 64 synthetic "\n'
        '                "zeros, KV canonicalized to fp8_ds_mla, backend "\n'
        '                "FLASHINFER_MLA_SPARSE_SM120"\n'
        "            )\n",
        "sm120-rope-pad",
    )
    # R3: Q zero-pad at kernel boundary (+ assert original width).
    text = replace_once(
        text,
        "        if isinstance(q, tuple):\n"
        "            q = torch.cat(q, dim=-1)\n"
        "\n"
        "        num_actual_toks = q.shape[0]\n",
        "        if isinstance(q, tuple):\n"
        "            q = torch.cat(q, dim=-1)\n"
        f"        if self.rope_pad:  # {MARK} model stays semantically NoPE\n"
        "            if q.shape[-1] != 512:\n"
        "                raise RuntimeError(\n"
        '                    "[SM121 MLA] expected NoPE query last dim 512 "\n'
        '                    f"before padding; got {q.shape[-1]}."\n'
        "                )\n"
        "            q = torch.nn.functional.pad(q, (0, self.rope_pad))\n"
        '            logger.info_once("[SM121 MLA] Q 512 -> 576 zero padded")\n'
        "\n"
        "        num_actual_toks = q.shape[0]\n",
        "sm120-q-pad",
    )
    # R4: kernel sees the padded rope dim, not semantic 0.
    text = replace_once(
        text,
        "            qk_rope_head_dim=self.qk_rope_head_dim,\n",
        "            qk_rope_head_dim=self.kernel_qk_rope_head_dim,\n",
        "sm120-kernel-rope",
    )
    # R5: valid counts + capacity + empty rows (reference).
    text = replace_once(
        text,
        "        topk_indices_physical = cast(\n"
        "            torch.Tensor,\n"
        "            triton_convert_req_index_to_global_index(\n"
        "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
        "                attn_metadata.block_table,\n"
        "                topk_indices,\n"
        "                BLOCK_SIZE=attn_metadata.block_size,\n"
        "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
        "            ),\n"
        "        )\n",
        "        topk_indices_physical, topk_lengths = cast(\n"
        "            tuple[torch.Tensor, torch.Tensor],\n"
        "            triton_convert_req_index_to_global_index(\n"
        "                attn_metadata.req_id_per_token[:num_actual_toks],\n"
        "                attn_metadata.block_table,\n"
        "                topk_indices,\n"
        "                BLOCK_SIZE=attn_metadata.block_size,\n"
        "                NUM_TOPK_TOKENS=topk_indices.shape[1],\n"
        "                return_valid_counts=True,\n"
        "            ),\n"
        "        )\n"
        f"        sparse_topk_capacity = topk_indices_physical.shape[1]  # {MARK}\n"
        "        empty_rows = topk_lengths == 0\n"
        "        topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(\n"
        "            empty_rows, 0\n"
        "        )\n"
        "        topk_lengths = topk_lengths.clamp(min=1)\n",
        "sm120-valid-counts",
    )
    # R6/R7: per-row seq_lens + capacity as template width (reference).
    text = replace_once(
        text,
        "            seq_lens=None,\n            max_seq_len=attn_metadata.topk_tokens,\n",
        "            seq_lens=topk_lengths,\n            max_seq_len=sparse_topk_capacity,\n",
        "sm120-seq-lens",
    )
    text = replace_once(
        text,
        "            sparse_mla_top_k=attn_metadata.topk_tokens,\n",
        "            sparse_mla_top_k=sparse_topk_capacity,\n",
        "sm120-topk-cap",
    )
    # R8: empty-row zeroing + do_kv_cache_update with synthetic k_pe (reference).
    text = replace_once(
        text,
        "        return out.squeeze(1), None\n",
        "        out = out.squeeze(1)\n"
        "        out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)\n"
        "        return out, None\n"
        "\n"
        "    def do_kv_cache_update(\n"
        "        self,\n"
        "        kv_c_normed: torch.Tensor,\n"
        "        k_pe: torch.Tensor,\n"
        "        kv_cache: torch.Tensor,\n"
        "        slot_mapping: torch.Tensor,\n"
        "        kv_cache_dtype: str,\n"
        "        k_scale: torch.Tensor,\n"
        "    ) -> None:\n"
        "        if self.rope_pad:\n"
        "            if k_pe.shape[-1] != 0:\n"
        "                raise RuntimeError(\n"
        '                    "[SM121 MLA] expected empty K_PE (dim 0) for NoPE "\n'
        '                    f"model; got {k_pe.shape[-1]}."\n'
        "                )\n"
        "            k_pe = k_pe.new_zeros((k_pe.shape[0], 1, self.rope_pad))\n"
        '            logger.info_once("[SM121 MLA] K_PE 0 -> 64 synthetic zeros")\n'
        "        super().do_kv_cache_update(\n"
        "            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale\n"
        "        )\n",
        "sm120-kv-update",
    )
    return text


def patch_buffer_width(text: str, rel: str) -> str:
    if MARK in text:
        return text
    return replace_once(
        text,
        "buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)",
        "buffer_width = topk_tokens  # [glm53-sm121-mla] SM120 template needs 2048",
        f"buffer-width:{rel}",
    )


def patch_sparse_block(text: str) -> str:
    if MARK in text:
        return text
    return replace_once(
        text,
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        return [64, 256]\n",
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        f"        return [64]  # {MARK} GLM_NSA/DSV3_2 PAGE_BLOCK_SIZE=64 only\n",
        "sm120-block-64",
    )


def patch_cuda_align(text: str) -> str:
    if MARK in text:
        return text
    return replace_once(
        text,
        "        return index_kpool * min(PAGED_MQA_PAGE_SIZES)\n",
        "        page_sizes = PAGED_MQA_PAGE_SIZES\n"
        "        capability = cls.get_device_capability()\n"
        "        if capability is not None and capability.major == 12:\n"
        "            page_sizes = tuple(p for p in page_sizes if p == 64)\n"
        f"        return index_kpool * min(page_sizes)  # {MARK} DeepGEMM sm_120: 64 only\n",
        "cuda-align",
    )


def patch_indexer(text: str) -> str:
    if "pool_ids[:, : select_k - 1]" in text:
        return text  # already trimmed (idempotent)
    for old, new, label in (
        (
            "                    expanded = expand_pools_and_append_tail(\n"
            "                        pool_ids, q_seq, index_kpool\n"
            "                    )\n",
            "                    expanded = expand_pools_and_append_tail(\n"
            f"                        pool_ids[:, : select_k - 1], q_seq, index_kpool  # {MARK}\n"
            "                    )\n",
            "indexer-prefill",
        ),
        (
            "            out = expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)\n",
            "            out = expand_pools_and_append_tail(\n"
            f"                pool_ids[:, : select_k - 1], dec_seq, index_kpool  # {MARK}\n"
            "            )\n",
            "indexer-decode",
        ),
    ):
        text = replace_once(text, old, new, label)
    return text


def verify(work: Path) -> None:
    import inspect  # noqa: F401  (parity with reference Dockerfile checks)

    sm120 = (work / "flashinfer_mla_sparse_sm120.py").read_text()
    for needle in (
        "supports_dense_mha_prefill = False",
        "self.rope_pad = 64",
        "self.kernel_qk_rope_head_dim",
        "torch.nn.functional.pad(q, (0, self.rope_pad))",
        "qk_rope_head_dim=self.kernel_qk_rope_head_dim",
        "return_valid_counts=True",
        "sparse_mla_top_k=sparse_topk_capacity",
        "seq_lens=topk_lengths",
        "out.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)",
        "def do_kv_cache_update",
        "[SM121 MLA]",
    ):
        if needle not in sm120:
            raise SystemExit(f"verify: missing in sm120.py: {needle!r}")
    if "attn_metadata.topk_tokens" in sm120:
        raise SystemExit("verify: stale attn_metadata.topk_tokens in sm120.py")
    for rel in ("models/glm5next/nvidia/model.py", "models/glm5next/nvidia/mtp.py"):
        name = "glm5next_model.py" if rel.endswith("model.py") else "glm5next_mtp.py"
        src = (work / name).read_text()
        if "buffer_width = topk_tokens  # [glm53-sm121-mla]" not in src:
            raise SystemExit(f"verify: buffer_width not patched in {name}")
        if "buffer_width = topk_tokens + (kpool" in src:
            raise SystemExit(f"verify: buffer_width not patched in {name}")
    kpool = (work / "sparse_attn_indexer_kpool.py").read_text()
    if kpool.count("pool_ids[:, : select_k - 1]") != 2:
        raise SystemExit("verify: indexer pool trim count != 2")
    sparse = (work / "flashinfer_mla_sparse.py").read_text()
    if "return [64]" not in sparse:
        raise SystemExit("verify: SM120 block size not pinned to [64]")
    cuda = (work / "cuda.py").read_text()
    if "if capability is not None and capability.major == 12:" not in cuda:
        raise SystemExit("verify: cuda.py SM12 alignment missing")
    for p in sorted(work.glob("*.py")):
        if p.name.startswith("pristine_"):
            continue
        compile(p.read_text(), str(p), "exec")
    print("glm53 SM121 MLA overlay verify OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--base-indexer", default="",
                    help="existing kpool-fixed indexer to preserve (flag=0 base)")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        verify(work)
        return 0

    pristine = extract_pristine(args.image, work)

    # Indexer base: preserve the currently-mounted SM121 persistent_topk fix
    # so flag=1 stays a strict superset of flag=0.
    indexer_base = pristine["indexer.py"]
    if args.base_indexer:
        base = Path(args.base_indexer)
        if base.is_file():
            cand = base.read_text()
            # sanity: must contain the trim anchors' context
            if "expand_pools_and_append_tail" in cand:
                indexer_base = cand
                print(f"[sm121] indexer base: {base} (keeps persistent_topk fix)")
            else:
                print(f"[sm121] WARN: {base} lacks indexer body; using pristine",
                      file=sys.stderr)
        else:
            print(f"[sm121] WARN: base-indexer {base} missing; using pristine",
                  file=sys.stderr)

    patched = {
        "flashinfer_mla_sparse_sm120.py": patch_sm120(pristine["sm120.py"]),
        "flashinfer_mla_sparse.py": patch_sparse_block(pristine["sparse.py"]),
        "cuda.py": patch_cuda_align(pristine["cuda.py"]),
        "sparse_attn_indexer_kpool.py": patch_indexer(indexer_base),
        "glm5next_model.py": patch_buffer_width(pristine["glm_model.py"], "model.py"),
        "glm5next_mtp.py": patch_buffer_width(pristine["glm_mtp.py"], "mtp.py"),
    }
    for name, text in patched.items():
        (work / name).write_text(text)
    verify(work)
    print(f"[sm121] wrote 6 patched files to {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
