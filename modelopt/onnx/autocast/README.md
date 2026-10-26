# modelopt/onnx/autocast

AutoCast: converts an FP32 ONNX model to mixed precision (FP16 or BF16). Nodes that are safe run in low precision. Nodes that can overflow or lose accuracy stay in FP32.

## Purpose

- Convert an FP32 ONNX model to `fp16` or `bf16` mixed precision (`convert_to_mixed_precision()`).
- Keep risky nodes in FP32 by rules: node-name regex, op type, initializer range (`init_max`), I/O value range from a reference run (`data_max`), depth of reduction, or a custom `NodeRuleBase` rule.
- Force nodes into low precision by node-name regex or op type (include rules).
- Run the model with ONNX Runtime (CPU, CUDA or TensorRT execution providers) to get reference I/O ranges, with random or user calibration data.
- Clean the graph before conversion: opset upgrade (default minimum 13 for `fp16`, 22 for `bf16`), LayerNorm pattern fusion, shared-constant duplication, custom-op handling, FP64 to FP32.
- Insert `Cast` nodes at precision borders and optionally keep the network I/O types (`keep_io_types`).
- Give a light conversion path for other ModelOpt flows (`convert_to_f16()`): a fixed op block list, no node classifier, no reference run.
- CLI: `python -m modelopt.onnx.autocast`.

## Inter: interaction with other systems

```mermaid
flowchart LR
    user["User / CLI"]:::ext -->|"convert_to_mixed_precision(onnx_path, ...)"| ac["modelopt/onnx/autocast"]:::own
    q["modelopt/onnx/quantization (precision_utils.py)"]:::int -->|"convert_to_f16(model, ...)"| ac
    dep["modelopt/torch/_deploy (torch_onnx.py)"]:::int -->|"convert_to_f16(model, ...)"| ac
    ac -->|"shared ONNX helpers"| onnxu["modelopt/onnx (utils.py, trt_utils.py)"]:::int
    ac -->|"reference run for I/O ranges"| ort["onnxruntime"]:::ext
    ac -->|"graph edits"| gs["onnx, onnx_graphsurgeon"]:::ext
    ac -->|"mixed-precision .onnx"| trt["TensorRT / ONNX Runtime"]:::ext
    classDef own fill:#d4edda,stroke:#2e7d32,color:#000
    classDef int fill:#cfe2ff,stroke:#1565c0,color:#000
    classDef ext fill:#fff3cd,stroke:#b8860b,color:#000
    linkStyle default stroke:#555
```

## Components

| Component | Purpose |
|-----------|---------|
| `convert.py` | Public API: `convert_to_mixed_precision()` (full flow) and `convert_to_f16()` (block-list flow, used by other modules) |
| `__main__.py` | CLI for `convert_to_mixed_precision()` |
| `graphsanitizer.py` | `GraphSanitizer`: opset and IR version, LayerNorm fusion, shared constants, custom ops, I/O casts, FP64 to FP32 |
| `nodeclassifier.py` | `NodeClassifier` and rules (`DisabledNodeNameRegexRule`, `DisabledOpTypes`, `InitializerRangeRule`, `IORangeRule`, `DepthOfReductionRule`, include rules): split nodes into low and high precision |
| `referencerunner.py` | `ReferenceRunner`: run the model with ONNX Runtime and collect output tensors or `TensorStats` |
| `precisionconverter.py` | `PrecisionConverter`: change initializers and value types, insert and clean up `Cast` nodes |
| `utils.py` | Graph mappings, subgraph walk, type and shape clearing, ops not supported in low precision |
| `logging_config.py` | `configure_logging()` for the AutoCast logger |

## Intra: interaction between components

```mermaid
sequenceDiagram
    participant U as User / __main__
    participant C as convert.py
    participant S as GraphSanitizer
    participant R as ReferenceRunner
    participant N as NodeClassifier
    participant P as PrecisionConverter
    U->>C: convert_to_mixed_precision(onnx_path, low_precision_type, ...)
    C->>S: sanitize(model)
    S-->>C: clean model, custom ops
    C->>C: infer types, load external data, setup_mappings
    opt data_max set or custom ops found
        C->>R: run(calibration_data)
        R-->>C: reference outputs
    end
    C->>N: run(reference outputs)
    N-->>C: low / high precision nodes
    C->>P: convert(high, low)
    P-->>C: mixed-precision ModelProto
    C-->>U: ModelProto (CLI saves .onnx)
```

`convert_to_f16()` also runs `GraphSanitizer`, but skips `ReferenceRunner` and `NodeClassifier`. It builds the node lists from `op_block_list` and `nodes_to_exclude`, then calls `PrecisionConverter`.

## Key integration tests

| Test | What it proves |
|------|----------------|
| `tests/unit/onnx/autocast/test_autocast.py` | `convert_to_mixed_precision()` end to end on sample models |
| `tests/unit/onnx/autocast/test_graphsanitizer.py` | Graph cleaning (LayerNorm fusion, opset, casts) |
| `tests/unit/onnx/autocast/test_nodeclassifier.py` | Each classification rule |
| `tests/unit/onnx/autocast/test_precisionconverter.py` | Cast insertion and type changes, `keep_io_types` |
| `tests/unit/onnx/autocast/test_referencerunner.py` | Reference run with random and file data |
| `tests/unit/onnx/test_autocast_quantize.py` | AutoCast together with ONNX quantization |

User guide: `docs/source/guides/8_autocast.rst`.
