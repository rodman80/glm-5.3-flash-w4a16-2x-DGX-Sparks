#!/usr/bin/env bash
#
# GLM-5.3-Flash W4A16-MTP + DFlash2, TP2 on 2x DGX Spark (GB10/SM121)
# Verbatim parameters from the HF README canada-quant/glm-5.3-w4a16-mtp (commit 4eeb77a)
# Worker (1) FIRST, then head (0). Image radixark/vllm-glm53-flash:sm121-v11-dflash2
#
# Usage: ./launch-glm53-w4a16-tp2-dflash2.sh <0|1>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ENV_FILE="${ENV_FILE:-.env}"  # ENV_FILE=.env.base for baseline A/B boots
[ -f "$SCRIPT_DIR/$ENV_FILE" ] || cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/$ENV_FILE"
set -a; source "$SCRIPT_DIR/$ENV_FILE"; set +a

NODE_RANK="${1:?usage: $0 <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || { echo "rank must be 0 or 1" >&2; exit 2; }

IMAGE="${IMAGE:-radixark/vllm-glm53-flash:sm121-v11-dflash2}"
IMAGE_FALLBACK="${IMAGE_FALLBACK:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}"
NAME="${NAME:-vllm_glm53_w4a16}"

MODEL_HOST_PATH="${MODEL_HOST_PATH:-/var/tmp/glm-5.3-flash-w4a16-mtp}"
MODEL_PATH="${MODEL_PATH:-/models/glm-5.3-flash-w4a16-mtp}"
DFLASH_HOST_PATH="${DFLASH_HOST_PATH:-/var/tmp/models/GLM-5.3-Flash-DFlash2}"
DFLASH_PATH="${DFLASH_PATH:-/models/dflash2-draft}"
CACHE_HOST_PATH="${CACHE_HOST_PATH:-/var/tmp/glm53-w4a16-cache}"

HEAD_IP="${HEAD_IP:-10.100.24.2}"
WORKER_IP="${WORKER_IP:-10.100.24.1}"
MPORT="${MASTER_PORT:-29521}"
PORT="${PORT:-8000}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
# No ':' on purpose: empty in .env = no pin (profiler sizes it).
# Only falls back to the HF pin when the var doesn't exist at all (old checkout).
KV_CACHE_MEMORY="${KV_CACHE_MEMORY-9663676416}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
BLOCK_SIZE="${BLOCK_SIZE:-2304}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
SPEC_METHOD="${SPEC_METHOD:-dflash}"
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
MTP_TOKENS="${MTP_TOKENS:-2}"
MOE_BACKEND="${MOE_BACKEND:-}"

HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f0np0}"
WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f0np0}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f0}"
WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
HEAD_GID="${HEAD_GID:-$NCCL_IB_GID_INDEX}"
WORKER_GID="${WORKER_GID:-$NCCL_IB_GID_INDEX}"

case "$NODE_RANK" in
  0) HOST_IP="$HEAD_IP"; CX7_IF="$HEAD_CX7_IF"; CX7_IB="$HEAD_CX7_IB"; GID="$HEAD_GID"; HEADLESS="" ;;
  1) HOST_IP="$WORKER_IP"; CX7_IF="$WORKER_CX7_IF"; CX7_IB="$WORKER_CX7_IB"; GID="$WORKER_GID"; HEADLESS="--headless" ;;
esac

# --- preflight ------------------------------------------------------------
test -f "$MODEL_HOST_PATH/config.json" || { echo "missing $MODEL_HOST_PATH/config.json — run ./download.sh" >&2; exit 2; }
grep -q '"quantization_config"' "$MODEL_HOST_PATH/config.json" || echo "WARN: quantization_config not found — image may not load W4A16" >&2
grep -q 'compressed-tensors' "$MODEL_HOST_PATH/config.json" && echo "[ok] compressed-tensors detected" || echo "WARN: not compressed-tensors"
if [ "$SPEC_METHOD" = "dflash" ]; then
  test -d "$DFLASH_HOST_PATH" || { echo "missing drafter $DFLASH_HOST_PATH — run ./download.sh" >&2; exit 2; }
fi
# vision chat template (ships in this repo as chat_template_mm.jinja — the HF
# checkpoint does not include it; ours comes via the Tony repo, see README Credits)
CHAT_TMPL="$MODEL_HOST_PATH/chat_template_mm.jinja"
if [ ! -f "$CHAT_TMPL" ]; then
  if [ -f "$SCRIPT_DIR/chat_template_mm.jinja" ]; then cp "$SCRIPT_DIR/chat_template_mm.jinja" "$CHAT_TMPL"; echo "[fix] chat_template_mm.jinja copied from repo"
  else echo "WARN: $CHAT_TMPL missing — image requests will 500" >&2; fi
fi

# Image: try primary, then fallback
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[pull] $IMAGE"
  docker pull "$IMAGE" || { echo "[fallback] $IMAGE_FALLBACK"; docker pull "$IMAGE_FALLBACK" && IMAGE="$IMAGE_FALLBACK"; } || true
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1 && docker image inspect "$IMAGE_FALLBACK" >/dev/null 2>&1; then
  IMAGE="$IMAGE_FALLBACK"
fi

mkdir -p "$CACHE_HOST_PATH" "$CACHE_HOST_PATH/root-cache" "$CACHE_HOST_PATH/torchinductor" "$CACHE_HOST_PATH/tilelang"

# kpool top-k SM121 patch (if present). Suppressed with GLM53_SM121_MLA=1:
# the SM120 overlay generates its own indexer (base = this same patch + trim),
# and two mounts on the same target would be ambiguous.
PATCH_ARGS=()
if [ "${GLM53_SM121_MLA:-0}" != "1" ] && [ -n "${PATCH_KPOOL_HOST:-}" ] && [ -f "$PATCH_KPOOL_HOST" ]; then
  PATCH_ARGS=(-v "$PATCH_KPOOL_HOST:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro")
  echo "[patch] SM121 sparse_attn_indexer_kpool mounted"
fi

# APC patch: hybrid prefix-cache zeroed by the drafter SWA group (stock vLLM
# marks every eagle group and the drafter SWA zeroes the hybrid min -> 0 hits).
# Source: overlay/patch_hybrid_prefix_hit.py from the GLM-5.3-Flash-EXL3 repo
# (same Glm5Next groups + DFlash2; anchors verified against our image).
# Generated per node from the image (agnostic), fail-closed: without markers,
# no mount — boot continues stock. Logs "hybrid APC groups:" at boot.
APC_ARGS=()
if [ "${APPLY_APC_PATCH:-1}" = "1" ]; then
  APC_SCRIPT="${PATCH_APC_SCRIPT:-$SCRIPT_DIR/docs/patch_hybrid_prefix_hit.py}"
  APC_WORK="${APC_WORK_DIR:-/var/tmp/glm53-w4a16-cache/apc}"
  mkdir -p "$APC_WORK"
  if [ -f "$APC_SCRIPT" ] && { [ ! -f "$APC_WORK/coordinator.patched.py" ] || ! grep -q "glm53-hybrid-apc" "$APC_WORK/coordinator.patched.py"; }; then
    _apc_cid="$(docker create "$IMAGE" 2>/dev/null)" || true
    if [ -n "${_apc_cid:-}" ]; then
      docker cp "$_apc_cid:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py" "$APC_WORK/coordinator.pristine.py" 2>/dev/null || true
      docker rm "$_apc_cid" >/dev/null 2>&1 || true
    fi
    if [ -f "$APC_WORK/coordinator.pristine.py" ]; then
      cp "$APC_WORK/coordinator.pristine.py" "$APC_WORK/coordinator.patched.py"
      GLM53_KV_COORDINATOR_PY="$APC_WORK/coordinator.patched.py" python3 "$APC_SCRIPT" 2>&1 | tail -n 1 || true
    fi
  fi
  if [ -f "$APC_WORK/coordinator.patched.py" ] && grep -q "glm53-hybrid-apc" "$APC_WORK/coordinator.patched.py"; then
    APC_ARGS=(-v "$APC_WORK/coordinator.patched.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py:ro")
    echo "[apc] hybrid prefix-hit patch mounted"
  else
    echo "WARN: APC not patched (anchors drifted?) — boot continues stock" >&2
  fi
fi

# SM120 sparse MLA overlay (GLM53_SM121_MLA=1): NoPE MLA -> packed fp8_ds_mla
# -> GLM_NSA 576-wide kernel via zero-pad (docs/patch_sm121_mla.py, port of the
# working EXL3 overlay; Marlin/W4A16 untouched — only attention/KV change).
# Generated per node from the image (fail-closed: any diverged anchor
# ABORTS the boot — a silent stock boot would invalidate the A/B).
# Surgical selection via backend_per_kind (only the mla_attention group changes;
# indexer/KDA/drafter/vision stay auto). 0 = SM90 baseline fully preserved.
SM121_MOUNTS=()
SM121_ARGS=()
if [ "${GLM53_SM121_MLA:-0}" = "1" ]; then
  SM121_SCRIPT="${SM121_PATCH_SCRIPT:-$SCRIPT_DIR/docs/patch_sm121_mla.py}"
  SM121_WORK="${SM121_WORK_DIR:-/var/tmp/glm53-w4a16-cache/sm121}"
  mkdir -p "$SM121_WORK"
  SM121_BASE_ARGS=()
  if [ -n "${PATCH_KPOOL_HOST:-}" ] && [ -f "$PATCH_KPOOL_HOST" ]; then
    SM121_BASE_ARGS=(--base-indexer "$PATCH_KPOOL_HOST")
  fi
  if python3 "$SM121_SCRIPT" --image "$IMAGE" --work-dir "$SM121_WORK" "${SM121_BASE_ARGS[@]}" 2>&1 | tail -n 3; then
    _sm121_site=/usr/local/lib/python3.12/dist-packages/vllm
    SM121_MOUNTS=(
      -v "$SM121_WORK/flashinfer_mla_sparse_sm120.py:$_sm121_site/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py:ro"
      -v "$SM121_WORK/flashinfer_mla_sparse.py:$_sm121_site/v1/attention/backends/mla/flashinfer_mla_sparse.py:ro"
      -v "$SM121_WORK/cuda.py:$_sm121_site/platforms/cuda.py:ro"
      -v "$SM121_WORK/sparse_attn_indexer_kpool.py:$_sm121_site/model_executor/layers/sparse_attn_indexer_kpool.py:ro"
      -v "$SM121_WORK/glm5next_model.py:$_sm121_site/models/glm5next/nvidia/model.py:ro"
      -v "$SM121_WORK/glm5next_mtp.py:$_sm121_site/models/glm5next/nvidia/mtp.py:ro"
    )
    SM121_ARGS=(--attention-config '{"backend_per_kind": {"mla_attention": "FLASHINFER_MLA_SPARSE_SM120"}}')
    echo "[sm121] overlay mounted (6 files) + mla_attention=FLASHINFER_MLA_SPARSE_SM120"
  else
    echo "[sm121] ERROR: overlay generation failed — aborting (see docs/patch_sm121_mla.py)" >&2
    exit 3
  fi
fi

# Optional MoE backend
MOE_ARGS=()
if [ -n "$MOE_BACKEND" ]; then MOE_ARGS=(--moe-backend "$MOE_BACKEND"); fi

# Async scheduling: sobrepoe scheduling de CPU com execucao da GPU.
# Default off in this build; 1 = --async-scheduling.
SCHED_ARGS=()
if [ "${ASYNC_SCHEDULING:-0}" = "1" ]; then SCHED_ARGS=(--async-scheduling); fi

# KV cache memory: empty = let the profiler size it (Tony's safe fallback)
KV_ARGS=()
if [ -n "${KV_CACHE_MEMORY:-}" ]; then KV_ARGS=(--kv-cache-memory "$KV_CACHE_MEMORY"); fi

# FlashInfer autotune always saved 0 configs on this W4A16-Marlin lane
# (verified on 2 boots) and costs ~50-85s per boot. 1 = skip with
# --no-enable-flashinfer-autotune (vLLM logs "Skipping FlashInfer autotune").
# Agnostic: any Spark on the default config can use it; to re-enable
# (other quant/model), set DISABLE_FLASHINFER_AUTOTUNE=0 in .env.
AUTOTUNE_ARGS=()
if [ "${DISABLE_FLASHINFER_AUTOTUNE:-0}" = "1" ]; then AUTOTUNE_ARGS=(--no-enable-flashinfer-autotune); fi

# speculative config (correct per method)
# SM120 (GLM53_SM121_MLA=1): the target canonicalizes the global cache to
# fp8_ds_mla, which no non-MLA backend accepts. The DFlash2 drafter inherits the
# global dtype (speculative kv_cache_dtype=None) and dies in backend
# selection (non-causal SWA + fp8_ds_mla). Force the drafter to "auto"
# (dense bf16 KV, same fix as the EXL3 ref); target stays fp8_ds_mla.
# Baseline (flag=0): JSON byte-identical to the stock one.
if [ "$SPEC_METHOD" = "dflash" ]; then
  if [ "${GLM53_SM121_MLA:-0}" = "1" ]; then
    SPEC_JSON="{\"method\":\"dflash\",\"model\":\"$DFLASH_PATH\",\"num_speculative_tokens\":$DFLASH_TOKENS,\"kv_cache_dtype\":\"auto\"}"
  else
    SPEC_JSON="{\"method\":\"dflash\",\"model\":\"$DFLASH_PATH\",\"num_speculative_tokens\":$DFLASH_TOKENS}"
  fi
  DFLASH_VOL=(-v "$DFLASH_HOST_PATH:$DFLASH_PATH:ro")
  SPEC_ARGS=(--speculative-config "$SPEC_JSON")
elif [ "$SPEC_METHOD" = "none" ]; then
  # No spec: validates the target in isolation (SM120 bring-up, kernel A/B).
  DFLASH_VOL=()
  SPEC_ARGS=()
  echo "[spec] none (no speculative decoding)"
else
  SPEC_JSON="{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}"
  DFLASH_VOL=()
  SPEC_ARGS=(--speculative-config "$SPEC_JSON")
fi

docker rm -f "$NAME" 2>/dev/null || true

# --- NCCL fabric (single default; DUAL_NIC=1 = validated dual-fabric) -------
# Validated 2026-09-07 (torch NCCL probe, same image):
#   single .24:  9.67 /  9.64 / 10.39 GB/s @ 64/256/1024MB
#   dual .24+.25: 19.7 / 17.5 / 20.8 GB/s (channels striped IB/0+IB/1)
# f1 ports are DOWN on these Sparks — UP 200G pair is f0 of both fabrics.
# GID_INDEX unset under DUAL: head .24 RoCEv2=idx4, worker .24 + both .25 =idx3
# (one index can't fit all -> NCCL auto-picks with ROCE_VERSION_NUM=2).
# ADDR_RANGE /16 required: /24 would exclude the .25 NIC entirely.
NCCL_HCA_ARGS=(-e NCCL_IB_HCA="$CX7_IB" -e NCCL_IB_GID_INDEX="$GID")
NCCL_RANGE_ARGS=(-e NCCL_IB_ADDR_RANGE=10.100.24.0/24)
NCCL_SOCK_ARGS=(-e NCCL_SOCKET_IFNAME="$CX7_IF")
NCCL_XNIC_ARGS=(-e NCCL_CROSS_NIC=0)
NCCL_QPS_ARGS=()
# Default 1 since 2026-09-07 (A/B showed no regression: decode unchanged,
# prefill +~5%); DUAL_NIC=0 restores single-NIC .24.
if [ "${DUAL_NIC:-1}" = "1" ]; then
  NCCL_HCA_ARGS=(-e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0)
  NCCL_RANGE_ARGS=(-e NCCL_IB_ADDR_RANGE=10.100.0.0/16)
  NCCL_SOCK_ARGS=(-e NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0)
  NCCL_XNIC_ARGS=(-e NCCL_CROSS_NIC=1)
  NCCL_QPS_ARGS=(-e NCCL_IB_QPS_PER_CONNECTION=4 -e NCCL_IB_SPLIT_DATA_ON_QPS=1)
  echo "[nccl] DUAL_NIC=1 (rocep1s0f0+roceP2p1s0f0, cross-nic, 4qp, nogid)"
fi

echo "[launch] rank=$NODE_RANK host=$HOST_IP head=$HEAD_IP GID=$GID image=$IMAGE W4A16-MTP $MAX_MODEL_LEN KV_MEM=${KV_CACHE_MEMORY:-profiler} spec=$SPEC_METHOD moe=${MOE_BACKEND:-auto} sm121=${GLM53_SM121_MLA:-0}"

set -x
docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -v "$CACHE_HOST_PATH/root-cache:/root/.cache" \
  "${DFLASH_VOL[@]}" \
  "${PATCH_ARGS[@]}" \
  "${APC_ARGS[@]}" \
  "${SM121_MOUNTS[@]}" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HOME=/cache/huggingface \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TILELANG_CACHE_DIR=/cache/tilelang \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  "${NCCL_HCA_ARGS[@]}" \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  "${NCCL_RANGE_ARGS[@]}" \
  "${NCCL_SOCK_ARGS[@]}" -e GLOO_SOCKET_IFNAME="$CX7_IF" \
  -e TP_SOCKET_IFNAME="$CX7_IF" -e MN_IF_NAME="$CX7_IF" \
  -e NCCL_NVLS_ENABLE=0 "${NCCL_XNIC_ARGS[@]}" -e NCCL_IB_MERGE_NICS=0 "${NCCL_QPS_ARGS[@]}" \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD:-0}" \
  -e VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}" \
  "$IMAGE" \
    "$MODEL_PATH" \
    --served-model-name "${SERVED_MODEL_NAME:-glm-5.3-flash}" \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" --block-size "$BLOCK_SIZE" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" "${KV_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${SCHED_ARGS[@]}" \
    "${AUTOTUNE_ARGS[@]}" \
    "${SM121_ARGS[@]}" \
    --enforce-eager --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    "${SPEC_ARGS[@]}" \
    --tool-call-parser glm47 --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --chat-template "$MODEL_PATH/chat_template_mm.jinja" \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS
set +x
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep -q "$NAME" || {
  echo "$NAME exited — logs:" >&2
  docker logs "$NAME" 2>&1 | tail -n 120 >&2
  exit 1
}
echo "OK $NAME rank=$NODE_RANK host=$HOST_IP GID=$GID — wait for /health (6-8 min cold)"
