#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
[ -f "$SCRIPT_DIR/.env" ] || cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
set -a; source "$SCRIPT_DIR/.env"; set +a

MODEL="${MODEL:-canada-quant/glm-5.3-w4a16-mtp}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/var/tmp/glm-5.3-flash-w4a16-mtp}"
DFLASH_MODEL="${DFLASH_MODEL:-incoai/GLM-5.3-Flash-DFlash2}"
DFLASH_HOST_PATH="${DFLASH_HOST_PATH:-/var/tmp/models/GLM-5.3-Flash-DFlash2}"
HF_BIN="${HF_BIN:-huggingface-cli}"

need_hf() {
  if command -v hf >/dev/null 2>&1; then HF_BIN="hf"
  elif command -v huggingface-cli >/dev/null 2>&1 && huggingface-cli --help 2>&1 | grep -q "download.*local-dir"; then HF_BIN="huggingface-cli"
  elif python3 -c "import huggingface_hub" 2>/dev/null; then HF_BIN="python3 -m huggingface_hub.commands.huggingface_cli"
  else echo "ERROR: install huggingface_hub (pip install -U huggingface_hub)" >&2; exit 1; fi
}

log() { printf '\033[1;36m[glm53-w4a16]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm53-w4a16]\033[0m WARN: %s\n' "$*" >&2; }

need_hf
log "HF bin: $HF_BIN"

# disk check first
avail=$(df -BG /var/tmp 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
if [ -n "$avail" ] && [ "$avail" -lt 190 ] 2>/dev/null; then
  warn "/var/tmp has ${avail}G free, needs ~190G (178G model + 2.3G drafter) — free space"
fi

# W4A16-MTP: ~178 GiB (9 shards + mtp + f32patch) + chat template
log "Downloading $MODEL → $MODEL_HOST_PATH ( ~178 GiB, may take 20-40 min )"
mkdir -p "$MODEL_HOST_PATH"
if [ -n "${MODEL_REVISION:-}" ]; then
  $HF_BIN download "$MODEL" --local-dir "$MODEL_HOST_PATH" --revision "$MODEL_REVISION" 2>&1 | tail -n 20 || \
  $HF_BIN download "$MODEL" --local-dir "$MODEL_HOST_PATH" 2>&1 | tail -n 20
else
  $HF_BIN download "$MODEL" --local-dir "$MODEL_HOST_PATH" 2>&1 | tail -n 20
fi

# drafter for DFlash2 (~2.3 GiB)
log "Downloading $DFLASH_MODEL → $DFLASH_HOST_PATH"
mkdir -p "$DFLASH_HOST_PATH"
$HF_BIN download "$DFLASH_MODEL" --local-dir "$DFLASH_HOST_PATH" 2>&1 | tail -n 10 || true

# multimodal chat template — not on HF, bundled in this repo (via the Tony repo, see README Credits)
if [ ! -f "$MODEL_HOST_PATH/chat_template_mm.jinja" ] && [ -f "$SCRIPT_DIR/chat_template_mm.jinja" ]; then
  cp "$SCRIPT_DIR/chat_template_mm.jinja" "$MODEL_HOST_PATH/chat_template_mm.jinja"
  log "chat_template_mm.jinja copied to $MODEL_HOST_PATH"
fi
if [ ! -f "$MODEL_HOST_PATH/chat_template_mm.jinja" ]; then
  warn "$MODEL_HOST_PATH/chat_template_mm.jinja missing — copy chat_template_mm.jinja from this repo"
fi

# summary
log "Shards: $(find "$MODEL_HOST_PATH" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l) (expected 11: 9 + mtp + f32patch)"
ls -lh "$MODEL_HOST_PATH"/model*.safetensors 2>/dev/null | head -n 20 || true
ls -lh "$MODEL_HOST_PATH"/config.json "$MODEL_HOST_PATH"/model.safetensors.index.json 2>/dev/null || true
if [ -f "$MODEL_HOST_PATH/config.json" ]; then
  python3 -c "import json; j=json.load(open('$MODEL_HOST_PATH/config.json')); print(f\"  quant: {j['quantization_config']['quant_method']} {j['quantization_config']['format']} | max_pos {j['text_config']['max_position_embeddings']} | layers {j['text_config']['num_hidden_layers']}\")" 2>/dev/null || true
fi
log "Drafter shards: $(find "$DFLASH_HOST_PATH" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)"
log "OK — weights in $MODEL_HOST_PATH and drafter in $DFLASH_HOST_PATH"
log "Next: rsync to the worker (automatic in ./start.sh) or manual: rsync -av $MODEL_HOST_PATH/ \$WORKER_SSH:$MODEL_HOST_PATH/"
