# modelopt/torch/export

Checkpoint export: writes a quantized (or pruned, or speculative-decoding) PyTorch model as a deployable checkpoint. The main format is the unified Hugging Face checkpoint (safetensors + `hf_quant_config.json` / `quantization_config`) that TensorRT-LLM, vLLM and SGLang load.

## Purpose

- Export Hugging Face `transformers` and `diffusers` models to a unified HF checkpoint (`export_hf_checkpoint()`): pack quantized weights, write scales, KV-cache scales and the quantization config.
- Export Megatron Core GPT models to an HF checkpoint (`export_mcore_gpt_to_hf()`), and import HF weights into Megatron Core (`import_mcore_gpt_from_hf()`), with per-model name mappings in `modelopt/torch/export/plugins/mcore_*.py`.
- Support the quantization formats of `modelopt.torch.quantization` (FP8, NVFP4, MXFP4/MXFP8, INT4 AWQ, INT8, W4A8, GGML/IQ formats and others; names in `quant_format.py`).
- Export large models in bounded memory: streaming export for offloaded and FSDP2-sharded models (`unified_export_hf_streaming.py`), and per-layer shards written during calibration (`layerwise_export.py`).
- Copy the source weights the model did not load (for example MTP heads) and non-weight files, so the checkpoint is the complete model.
- Export vLLM fake-quant checkpoints (HF and Megatron) and speculative-decoding draft models (EAGLE, DFlash) in their official formats.
- Keep the deprecated TensorRT-LLM checkpoint export in `trtllm/` (`export_tensorrt_llm_checkpoint()`, removal planned in 0.49.0).

## Inter: interaction with other systems

```mermaid
flowchart LR
    user["examples (hf_ptq, megatron_bridge, diffusers, speculative_decoding)"]:::ext -->|"export_hf_checkpoint() / export_mcore_gpt_to_hf()"| ex["modelopt/torch/export"]:::own
    ex -->|"quantizers, formats, amax"| q["modelopt/torch/quantization"]:::int
    ex -->|"HF checkpoint files, state dict, distributed"| u["modelopt/torch/utils"]:::int
    ex -->|"model specs by model_type"| m["modelopt/torch/models"]:::int
    ex -->|"save modelopt_state"| opt["modelopt/torch/opt"]:::int
    ex -->|"read models / configs"| hf["transformers, diffusers, accelerate"]:::ext
    ex -->|"Megatron models"| mc["megatron-core"]:::ext
    ex -->|"safetensors + quant config"| rt["TensorRT-LLM, vLLM, SGLang"]:::ext
    classDef own fill:#d4edda,stroke:#2e7d32,color:#000
    classDef int fill:#cfe2ff,stroke:#1565c0,color:#000
    classDef ext fill:#fff3cd,stroke:#b8860b,color:#000
    linkStyle default stroke:#555
```

## Components

| Component | Purpose |
|-----------|---------|
| `unified_export_hf.py` | `export_hf_checkpoint()`: pick the path (diffusers, layerwise, streaming, FSDP2, in-memory), pack weights, write config; `export_speculative_decoding()` |
| `unified_export_hf_streaming.py` | Streaming export for offloaded and FSDP2 models: one layer at a time, then name shards and write the index |
| `layerwise_export.py` | `LayerwiseExporter`: write each decoder layer's shard as soon as it is calibrated, `finalize()` writes the rest |
| `unified_export_megatron.py` | `GPTModelExporter`, `export_mcore_gpt_to_hf()`, `import_mcore_gpt_from_hf()` |
| `registry.py` | `ExportModuleRegistry`, `PrepareMoEInputsRegistry`, `ExportContext`: dispatch per-module export logic |
| `hf_export_handlers.py` | Built-in handlers registered in `registry.py` (quantized linear, fused and MoE experts) |
| `quant_utils.py` | Scaling factors, quantization format of a module, weight packing (`to_quantized_weight()`), `postprocess_state_dict()` |
| `quantized_weight_export.py` | Functional export of quantized weight tensors and the HF quantization config |
| `quant_aware_conversion.py` | Quantization-aware reverse weight conversion (model layout back to HF layout) |
| `quant_format.py` | Quantization and KV-cache format names |
| `layer_utils.py` | Module-shape predicates and MoE quantizer helpers |
| `moe_utils.py` | MoE export helpers |
| `model_utils.py` | Model type detection |
| `convert_hf_config.py` | Convert the ModelOpt quant config to the llm-compressor format |
| `shard_cast_utils.py` | Shared primitives for streaming casts and shard writing |
| `diffusers_utils.py` | Diffusers component handling, NVFP4 padding and scale swizzle, per-layer metadata |
| `transformer_engine.py` | `convert_to_transformer_engine()` |
| `modelopt/torch/export/plugins/mcore_*.py`, `modelopt/torch/export/plugins/megatron_importer.py` | HF to Megatron Core name mappings per model family, and the importer |
| `modelopt/torch/export/plugins/vllm_fakequant_hf.py`, `modelopt/torch/export/plugins/vllm_fakequant_megatron.py` | vLLM fake-quant checkpoint export |
| `modelopt/torch/export/plugins/hf_spec_export.py`, `modelopt/torch/export/plugins/hf_spec_configs.py` | Speculative-decoding draft export and config templates |
| `trtllm/` | Deprecated TensorRT-LLM checkpoint export (`model_config_export.py` and helpers) |

## Intra: interaction between components

```mermaid
sequenceDiagram
    participant U as Caller
    participant H as unified_export_hf
    participant L as LayerwiseExporter
    participant S as unified_export_hf_streaming
    participant R as registry + hf_export_handlers
    participant Q as quant_utils + quantized_weight_export
    U->>H: export_hf_checkpoint(model, dtype, export_dir)
    H->>H: read unplaced source weights (extra_state_dict)
    alt layerwise exporter attached
        H->>L: finalize(extra_state_dict)
    else diffusers model
        H->>H: _export_diffusers_checkpoint()
    else offloaded or FSDP2-sharded
        H->>S: streaming export (one layer at a time)
        S->>R: per-module handlers
    else in memory
        H->>R: per-module handlers (ExportContext)
    end
    R->>Q: scales, pack weights, quant config
    Q-->>H: state dict + hf_quant_config
    H->>H: postprocess_state_dict, write safetensors, config, non-weight files
    H-->>U: checkpoint in export_dir
```

## Key integration tests

| Test | What it proves |
|------|----------------|
| `tests/unit/torch/export/test_unified_export_hf.py` | HF export on small models, CPU |
| `tests/unit/torch/export/test_export_registry.py` | Handler dispatch and precedence |
| `tests/unit/torch/export/test_quantized_weight_state.py` | Functional weight export state |
| `tests/unit/torch/export/test_quant_aware_conversion.py` | Reverse weight conversion |
| `tests/gpu/torch/export/test_unified_hf_export_and_check_safetensors.py` | Exported safetensors per format |
| `tests/gpu/torch/export/test_offload_export.py`, `test_fsdp2_export.py` | Streaming export for offloaded and FSDP2 models |
| `tests/gpu/torch/export/test_layerwise_export.py` | Per-layer export during calibration |
| `tests/gpu/torch/export/test_export_carry_over.py` | Unloaded source weights carried into the checkpoint |
| `tests/gpu/torch/export/test_export_diffusers.py` | Diffusers export |
| `tests/gpu_megatron/torch/export/test_unified_export_megatron.py` | Megatron Core export to HF |
| `tests/gpu_megatron/torch/export/test_vllm_fakequant_megatron_export.py` | vLLM fake-quant export from Megatron |
| `tests/gpu_trtllm/torch/export/test_export_compressed_nvfp4.py` | NVFP4 checkpoint loads in TensorRT-LLM |
