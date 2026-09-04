#!/usr/bin/env bash
# check-dflash2.sh — boot-signature gate for W4A16-MTP + DFlash2 (head).
# Ported from glm53-redhat-nvfp4-dgx-spark to canada-quant/glm-5.3-w4a16-mtp.
set -uo pipefail
NAME="${NAME:-vllm_glm53_w4a16}"
LOGFILE="$(mktemp)"
trap 'rm -f "$LOGFILE"' EXIT
docker logs "$NAME" > "$LOGFILE" 2>&1

declare -a SIGS=(
  "Using CompressedTensorsWNA16MoEMethod"
  "Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)"
  "Resolved architecture: DFlash2DraftModel"
  "Draft model DFlash2Qwen3ForCausalLM"
)

fail=0
for sig in "${SIGS[@]}"; do
  if grep -qF "$sig" "$LOGFILE"; then
    echo "OK   : $sig"
  else
    echo "MISS : $sig"
    fail=1
  fi
done

if grep -E "DFlash2(Qwen3ForCausalLM|Speculator|DraftModel)" "$LOGFILE" | grep -q .; then
  echo "OK   : DFlash2 classes active (not DFlash1)"
else
  echo "MISS : DFlash2 classes active"
  fail=1
fi

if grep -E "Traceback|AssertionError|NV_ERR_NO_MEMORY|out of bounds|EngineCore failed|KeyError" "$LOGFILE" | grep -q .; then
  echo "FAIL : hard error found in log:"
  grep -E "Traceback|AssertionError|NV_ERR_NO_MEMORY|out of bounds|EngineCore failed|KeyError" "$LOGFILE" | head -5
  fail=1
else
  echo "OK   : no hard errors in log"
fi

BASE="${BASE:-http://127.0.0.1:8000}"
if curl -sf --max-time 5 "$BASE/v1/models" > /dev/null; then
  echo "OK   : API serving on $BASE"
else
  echo "MISS : API not up yet (boot may still be in progress)"
  fail=1
fi

exit $fail
