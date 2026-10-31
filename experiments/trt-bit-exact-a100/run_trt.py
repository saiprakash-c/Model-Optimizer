# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build (optional) and run TensorRT engines, and save raw outputs (run on the GPU pod).

Usage:
  python run_trt.py build <work_dir>        # ONNX -> 4 engines in <work_dir>/engines
  python run_trt.py run <work_dir> <tag>    # run every engine RUNS times -> <work_dir>/out/<tag>/
"""

import hashlib
import json
import subprocess  # nosec B404
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

MODELS = ["resnet50", "dinov2_vitb14"]
PRECISIONS = {"fp32": [], "fp16": ["--fp16"]}  # fp32 keeps TensorRT's default TF32
RUNS = 5


def sha256(path):
    """Hex sha256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(work):
    """Build one engine per model and precision with trtexec."""
    (work / "engines").mkdir(exist_ok=True)
    for model in MODELS:
        for prec, flags in PRECISIONS.items():
            engine = work / "engines" / f"{model}_{prec}.engine"
            cmd = [
                "trtexec",
                f"--onnx={work}/onnx/{model}.onnx",
                f"--saveEngine={engine}",
                "--skipInference",
                *flags,
            ]
            log = work / "engines" / f"{model}_{prec}.build.log"
            with open(log, "w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)  # nosec B603 B607
            print(f"built {engine.name} {sha256(engine)[:16]}")


def run(work, tag):
    """Run each engine RUNS times; save the first output and every run's hash."""
    out = work / "out" / tag
    out.mkdir(parents=True, exist_ok=True)
    x = np.load(work / "inputs" / "input.npy")
    smi = subprocess.run(  # nosec B603 B607 - fixed nvidia-smi query
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,uuid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    meta = {"tag": tag, "gpu": smi, "tensorrt": trt.__version__, "engines": {}}
    for engine in sorted((work / "engines").glob("*.engine")):
        name = engine.stem
        hashes = []
        with TrtRunner(EngineFromBytes(BytesFromPath(str(engine)))) as runner:
            for i in range(RUNS):
                y = runner.infer({"input": x}, copy_outputs_to_host=True)["output"]
                y = np.ascontiguousarray(y)
                if i == 0:
                    np.save(out / f"{name}.npy", y)
                hashes.append(hashlib.sha256(y.tobytes()).hexdigest())
        meta["engines"][name] = {
            "engine_sha256": sha256(engine),
            "output_sha256_per_run": hashes,
            "dtype": str(y.dtype),
            "shape": list(y.shape),
        }
        print(f"{name}: {len(set(hashes))} distinct output hash(es) in {RUNS} runs")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "engines"}))


if __name__ == "__main__":
    cmd, work = sys.argv[1], Path(sys.argv[2])
    build(work) if cmd == "build" else run(work, sys.argv[3])
