#!/usr/bin/env bash
# validate.sh — gates for the HF repo canada-quant/glm-5.3-w4a16-mtp
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" 2>/dev/null && set +a || true
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/var/tmp/glm-5.3-flash-w4a16-mtp}"
DFLASH_HOST_PATH="${DFLASH_HOST_PATH:-/var/tmp/models/GLM-5.3-Flash-DFlash2}"
fail=0
ok() { echo "✓ $*"; }
bad() { echo "✗ $*"; fail=1; }
warn() { echo "  WARN: $*"; }

echo "== GLM-5.3 W4A16-MTP validation — canada-quant/glm-5.3-w4a16-mtp =="
echo "   HEAD_IP=${HEAD_IP:-?} WORKER_IP=${WORKER_IP:-?} GID head=${HEAD_GID:-?} worker=${WORKER_GID:-?}"
echo ""

# 0. Cluster checks
echo "--- cluster ---"
ip -br a 2>/dev/null | grep -E 'enp1s0f0|rocep' | sed 's/^/  /' || true
for i in 2 3; do printf "  GID %d head: " "$i"; cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/$i 2>/dev/null | tr -d '\n'; echo; done
sw=$(cat /proc/sys/vm/swappiness 2>/dev/null || echo "?")
[ "$sw" = "?" ] && warn "swappiness unreadable" || { [ "$sw" -le 10 ] 2>/dev/null && ok "swappiness=$sw" || warn "swappiness=$sw (ideal 0-10 for the 180G load)"; }
avail=$(df -BG /var/tmp 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
[ -n "$avail" ] && { [ "$avail" -ge 190 ] 2>/dev/null && ok "disk /var/tmp ${avail}G free" || warn "disk ${avail}G free, needs 190G"; } || true
echo ""

# 1. config exists
echo "--- model ---"
[ -f "$MODEL_HOST_PATH/config.json" ] && ok "config.json present ($MODEL_HOST_PATH)" || bad "config.json missing ($MODEL_HOST_PATH) — run ./download.sh"

# 2. quantization_config
if [ -f "$MODEL_HOST_PATH/config.json" ]; then
  grep -q 'pack-quantized' "$MODEL_HOST_PATH/config.json" && ok "pack-quantized" || bad "pack-quantized not found"
  grep -q '"num_bits": 4' "$MODEL_HOST_PATH/config.json" && ok "num_bits=4" || bad "num_bits!=4"
  grep -q '"group_size": 128' "$MODEL_HOST_PATH/config.json" && ok "group_size=128" || bad "group_size!=128"
  grep -q '"symmetric": true' "$MODEL_HOST_PATH/config.json" && ok "symmetric=true" || bad "symmetric not true"
  grep -q 'compressed-tensors' "$MODEL_HOST_PATH/config.json" && ok "compressed-tensors" || bad "quant_method is not compressed-tensors"
  grep -q 'model.visual' "$MODEL_HOST_PATH/config.json" && ok "ignore covers visual (BF16)" || bad "ignore without visual"
  python3 -c "import json; j=json.load(open('$MODEL_HOST_PATH/config.json')); assert any('layers.45' in x or 'layers\\\\.45' in x for x in j['quantization_config']['ignore'])" 2>/dev/null && ok "ignore covers layers.45 (MTP BF16)" || bad "ignore without MTP"
  python3 -c "import json; j=json.load(open('$MODEL_HOST_PATH/config.json')); assert j['text_config']['max_position_embeddings']==1048576" 2>/dev/null && ok "max_position_embeddings=1048576 (1M)" || bad "max_position_embeddings !=1M"
  python3 -c "import json; j=json.load(open('$MODEL_HOST_PATH/config.json')); assert j['text_config']['num_hidden_layers']==45" 2>/dev/null && ok "num_hidden_layers=45" || bad "num_hidden_layers !=45"
  python3 -c "import json; j=json.load(open('$MODEL_HOST_PATH/config.json')); assert j['text_config']['n_routed_experts']==288" 2>/dev/null && ok "n_routed_experts=288" || bad "n_routed_experts"
  python3 -c "import json; j=json.load(open('$MODEL_HOST_PATH/config.json')); assert j['text_config']['n_shared_experts']==1" 2>/dev/null && ok "n_shared_experts=1 / top8" || warn "n_shared_experts check"
fi

# 3. shards
if [ -d "$MODEL_HOST_PATH" ]; then
  n=$(find "$MODEL_HOST_PATH" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l)
  [ "$n" -ge 11 ] && ok "safetensors shards=$n (expected 11: 9+mtp+f32patch)" || bad "shards=$n expected 11"
  [ -f "$MODEL_HOST_PATH/model-mtp-00001.safetensors" ] && ok "model-mtp-00001.safetensors (14.8G BF16)" || bad "model-mtp missing"
  [ -f "$MODEL_HOST_PATH/model-f32patch-00001.safetensors" ] && ok "model-f32patch (6.7M)" || bad "f32patch missing"
  sz=$(du -sh "$MODEL_HOST_PATH" 2>/dev/null | cut -f1)
  echo "  total size: $sz (expected ~178 GiB)"
  ls -lh "$MODEL_HOST_PATH"/model.safetensors.index.json 2>/dev/null | awk '{print "  index:", $9, $5}' || true
fi
echo ""

# 4. generation
echo "--- generation ---"
if [ -f "$MODEL_HOST_PATH/generation_config.json" ]; then
  grep -q '"temperature": 1.0' "$MODEL_HOST_PATH/generation_config.json" && ok "generation_config temperature=1.0 top_p=0.95" || warn "generation_config off the HF default"
else warn "generation_config.json missing (uses model default)"; fi
echo ""

# 5. drafter
echo "--- drafter ---"
if [ -d "$DFLASH_HOST_PATH" ]; then
  dn=$(find "$DFLASH_HOST_PATH" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)
  ok "drafter $DFLASH_HOST_PATH present ($dn shards, ~2.3G)"
  ls -lh "$DFLASH_HOST_PATH"/config.json 2>/dev/null | awk '{print "  drafter config:", $9}' || true
else warn "drafter not downloaded — needed for DFlash2 spec-decode"; fi
echo ""

# 6. chat template
echo "--- vision ---"
[ -f "$MODEL_HOST_PATH/chat_template_mm.jinja" ] && ok "chat_template_mm.jinja present (vision)" || { warn "chat_template_mm.jinja missing in $MODEL_HOST_PATH — copy it from this repo (else vision 500s)"; ls -lh "$SCRIPT_DIR/chat_template_mm.jinja" 2>/dev/null | awk '{print "  repo has:", $9, $5}' || true; }
echo ""

# 7. image
echo "--- image ---"
_img="${IMAGE:-radixark/vllm-glm53-flash:sm121-v11-dflash2}"
if docker image inspect "$_img" >/dev/null 2>&1; then ok "image $_img local ($(docker images --format '{{.Size}}' "$_img" 2>/dev/null | head -n1))"; else warn "image not pulled — run docker pull $_img"; fi
_fallback="${IMAGE_FALLBACK:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}"
if [ "$_img" != "$_fallback" ] && docker image inspect "$_fallback" >/dev/null 2>&1; then ok "fallback $_fallback also local"; fi
if [ -n "${PATCH_KPOOL_HOST:-}" ] && [ -f "${PATCH_KPOOL_HOST:-}" ]; then ok "SM121 kpool patch $PATCH_KPOOL_HOST present"; else warn "kpool patch not found at ${PATCH_KPOOL_HOST:-\$PATCH_KPOOL_HOST} (needed on images < v11)"; fi
if [ "${GLM53_SM121_MLA:-0}" = "1" ]; then ok "SM120 overlay ACTIVE (docs/patch_sm121_mla.py + backend_per_kind mla_attention)"; else echo "  SM120 overlay off (SM90 baseline) — enable with GLM53_SM121_MLA=1"; fi
echo ""

# 8. network
echo "--- NCCL network ---"
echo "  HEAD ${HEAD_IP:-10.100.24.2}/${HEAD_CX7_IF:-enp1s0f0np0}/${HEAD_CX7_IB:-rocep1s0f0} GID ${HEAD_GID:-3}"
echo "  WORKER ${WORKER_IP:-10.100.24.1}/${WORKER_CX7_IF:-enp1s0f0np0}/${WORKER_CX7_IB:-rocep1s0f0} GID ${WORKER_GID:-3}"
ssh -o ConnectTimeout=5 "${WORKER_SSH:-${WORKER_IP:-10.100.24.1}}" "echo '  worker reachable: \$(hostname) \$(ip -br a | grep 10.100.24)'; for i in 2 3; do printf \"  worker GID \$i: \"; cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/\$i 2>/dev/null | tr -d \"\\n\"; echo; done" 2>&1 | sed 's/^/  /' || warn "worker ssh failed (${WORKER_SSH:-$WORKER_IP})"
echo ""

# 9. conflicting containers
echo "--- containers ---"
docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep -E 'vllm|glm' | sed 's/^/  head: /' || echo "  head: no vllm running"
ssh -o ConnectTimeout=5 "${WORKER_SSH:-${WORKER_IP:-10.100.24.1}}" "docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep -E 'vllm|glm' | sed 's/^/  worker: /' || echo '  worker: no vllm'" 2>&1 | sed 's/^/  /' || true
echo ""

if [ "$fail" -eq 0 ]; then echo "== ALL GATES PASSED (warnings ok) =="; else echo "== FAILURES FOUND — fix before ./start.sh =="; fi
exit "$fail"
