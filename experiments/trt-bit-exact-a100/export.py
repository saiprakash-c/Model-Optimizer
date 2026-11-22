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

"""Export ResNet-50 and DINOv2 ViT-B/14 to ONNX and write fixed inputs (run on CPU).

Usage: python export.py <work_dir>
Writes <work_dir>/onnx/{resnet50,dinov2_vitb14}.onnx and <work_dir>/inputs/input.npy.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from transformers import Dinov2Model

BATCH, SIZE, OPSET = 8, 224, 17


class Dinov2Wrapper(torch.nn.Module):
    """Return the last hidden state only, so the ONNX graph has one output."""

    def __init__(self):
        """Load facebook/dinov2-base with eager attention (exports to plain ONNX ops)."""
        super().__init__()
        self.model = Dinov2Model.from_pretrained(
            "facebook/dinov2-base", attn_implementation="eager"
        )

    def forward(self, pixel_values):
        """Return the last hidden state."""
        return self.model(pixel_values=pixel_values).last_hidden_state


def main(work):
    """Write the input and the two ONNX models to <work>."""
    (work / "onnx").mkdir(parents=True, exist_ok=True)
    (work / "inputs").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((BATCH, 3, SIZE, SIZE), dtype=np.float32)
    np.save(work / "inputs" / "input.npy", x)

    models = {
        "resnet50": torchvision.models.resnet50(weights="IMAGENET1K_V2"),
        "dinov2_vitb14": Dinov2Wrapper(),
    }
    for name, model in models.items():
        model.eval()
        torch.onnx.export(
            model,
            (torch.from_numpy(x),),
            work / "onnx" / f"{name}.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=OPSET,
            dynamo=False,
        )
        print(f"exported {name}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
