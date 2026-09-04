# Verbatim parameters — canada-quant/glm-5.3-w4a16-mtp

Snapshot: `4eeb77a3499fc2503290197118102eae2ed44553` (2026-09-02T23:23:31Z)
URL: https://huggingface.co/canada-quant/glm-5.3-w4a16-mtp

## config.json (essencial)

```json
{
  "architectures": ["Glm5NextForConditionalGeneration"],
  "model_type": "glm5_next",
  "dtype": "bfloat16",
  "tie_word_embeddings": false,
  "transformers_version": "5.16.1",
  "text_config": {
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 45,
    "num_nextn_predict_layers": 1,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "n_routed_experts": 288,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "vocab_size": 154880,
    "max_position_embeddings": 1048576,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_head_dim": 256,
    "v_head_dim": 256,
    "index_topk": 2048,
    "index_kpool": 4
  }
}
```

### quantization_config

- format: `pack-quantized` (compressed-tensors 0.18.0)
- quant_method: `compressed-tensors`
- group_0: targets=[Linear], weights {num_bits:4, group_size:128, symmetric:true, strategy:group, type:int, actorder:static, observer:memoryless_minmax}
- ignore: 124 + `re:model.language_model.layers.45..*` + lm_head — ver config.json integral (visual 24 blocos + shared_experts + self_attn + indexer + gate + layers 0-2 mlp + hc_*)
- Only 36,288 routed-expert GEMMs quantized; rest BF16

## generation_config.json

```json
{"_from_model_config": true, "do_sample": true, "temperature": 1.0, "top_p": 0.95, "eos_token_id": [154820,154827,154829], "pad_token_id": 154820}
```

## recipe.yaml

```yaml
default_stage:
  default_modifiers:
    GPTQModifier:
      targets: [Linear]
      ignore: [lm_head, 're:.*embed_tokens.*', 're:model.language_model\.layers\.45\..*',
        're:.*\.mlp\.gate$', 're:.*self_attn.*', 're:.*indexer.*', 're:.*shared_experts.*',
        're:^model.language_model\.layers\.[0-2]\.mlp\.(gate_proj|up_proj|down_proj)$', 're:.*visual.*',
        're:.*\.(attn_hc|ffn_hc)\..*', 're:.*hc_(attn|ffn)_.*']
      scheme: W4A16
      bypass_divisibility_checks: false
      requires_calibration_data: true
      block_size: 128
      dampening_frac: 0.01
      actorder: static
      offload_hessians: false
```

## Files in the repo

- model-00001..00009.safetensors (~20 GiB cada, total ~172 GiB)
- model-mtp-00001.safetensors (14.8 GiB, 889 keys, BF16)
- model-f32patch-00001.safetensors (6.7 MiB, hc_*/A_log/dt_bias/e_score_correction_bias)
- model.safetensors.index.json
- config.json (51 KiB)
- tokenizer.json (20 MiB), processor_config.json, chat_template.jinja

## Derived DGX Spark recipe

- Imagem: radixark/vllm-glm53-flash:sm121-v11-dflash2
- Drafter: incoai/GLM-5.3-Flash-DFlash2 (DFlash2 block-diffusion, layers [5,14,24,33,42], block 8, top_k 16, rank 256)
- 1M: MAX_MODEL_LEN 1048576 / KV_CACHE_MEM 9663676416 (1.36M nominal, ~1.34M measured) | 262K: 262144/3221225472
- max-num-seqs 6, block-size 2304, kv fp8_e4m3, enforce-eager, spec K=7 fixed, gpu-mem 0.85, timeout 3600
