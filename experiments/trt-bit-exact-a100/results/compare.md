a100_40gb: NVIDIA A100-SXM4-40GB, 40960 MiB, 570.148.08, GPU-87a03dd1-46ba-76cb-dc78-955b5578e999 | TensorRT 10.7.0
a100_80gb: NVIDIA A100-SXM4-80GB, 81920 MiB, 595.91.07, GPU-12f0d03e-64f0-35ea-664f-9aedc4043308 | TensorRT 10.7.0

| engine | runs equal on a100_40gb | runs equal on a100_80gb | a100_40gb vs a100_80gb |
|---|---|---|---|
| dinov2_vitb14_fp16 | yes (5 runs) | yes (5 runs) | bit-exact |
| dinov2_vitb14_fp32 | yes (5 runs) | yes (5 runs) | bit-exact |
| resnet50_fp16 | yes (5 runs) | yes (5 runs) | bit-exact |
| resnet50_fp32 | yes (5 runs) | yes (5 runs) | bit-exact |

engines compared: 4
RESULT: BIT-EXACT
