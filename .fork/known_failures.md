# Known failures on Thor (not bugs to fix)

From Thor test sweeps (2026-10-18). Do not spend budget on these unless the task is about them.

| Failure | Cause | What to do |
|---------|-------|-----------|
| CUDA/Triton extension build fails in `~/.venvs/mo` | Thor host has no nvcc, g++, or Python.h | Run GPU tests in the `mo-dev` container only (`gpu run` does this). |
| Extension build times out under pytest (120 s) | Each extension takes ~2 min to build on Thor | Precompile first: `python -c "from modelopt.torch.quantization.extensions import precompile; precompile()"`. |
| Later builds hang | A killed build left `~/.cache/torch_extensions/*/<ext>/lock` | Remove the stale lock. |
| ONNX tests that need CUDA or TensorRT execution providers fail (fp16 QLinear scales, missing TensorrtExecutionProvider) | aarch64 installs CPU-only onnxruntime | Expected on Thor. Verify on x86 (rented GPU) if needed. |
| `tests/gpu/onnx/test_trt_utils.py` segfaults after other ONNX tests | Not root-caused. Passes alone. | Run it alone. |
| DeepSpeed tests fail: NVML `nvmlDeviceGetMemoryInfo` NotSupported | DeepSpeed calls an NVML function that Thor does not support | Expected on Thor. |
| Container path must match host path | A different mount path breaks Triton and pytest (stale `__pycache__`) | `gpu run` mounts checkouts at the same path. |
