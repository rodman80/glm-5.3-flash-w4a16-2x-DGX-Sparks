#!/usr/bin/env bash
# Build a local profiling image; production launcher remains untouched.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";R="$(dirname "$D")";cd "$R"
BASE_IMAGE="${KB1_BASE_IMAGE:-radixark/vllm-glm53-flash:sm121-v11-dflash2}"
TAG="${KB1_IMAGE_TAG:-glm53-w4a16:kb1-moe-profile}"
WORK="${KB1_WORK_DIR:-/var/tmp/glm53-kb1-profile}"
rm -rf "$WORK";mkdir -p "$WORK"
python3 docs/patch_kb_moe_profile.py --image "$BASE_IMAGE" --work-dir "$WORK"
cat >"$WORK/Dockerfile" <<EOF
FROM $BASE_IMAGE
COPY marlin_moe.kb_profile.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py
ENV GLM53_KB_MOE_PROFILE=1 GLM53_KB_MOE_SAMPLE_EVERY=64 GLM53_KB_MOE_MAX_LOGS=2000
EOF
docker build -t "$TAG" "$WORK"
echo "Built $TAG"
echo "For the KB1 boot only, set IMAGE=$TAG in a temporary env file, restart, then run:"
echo "  ./bench/kb1_collect_moe_profile.sh"
