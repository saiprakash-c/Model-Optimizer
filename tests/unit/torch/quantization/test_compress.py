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

"""CPU tests of mtq.compress."""

import pytest
import torch
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.compress import is_real_quantized
from modelopt.torch.quantization.nn import GroupedQuantizer


@pytest.mark.parametrize("grouped", [False, True], ids=["linear", "grouped"])
def test_compress_refuses_w4a8_weights_before_changing_the_model(grouped):
    """A compressed W4A8 (SequentialQuantizer) weight can neither run forward nor be restored."""
    model = nn.Sequential(nn.Linear(128, 128, bias=False))  # 128 wide for W4A8's INT4 blocks
    mtq.quantize(model, mtq.W4A8_AWQ_BETA_CFG, lambda m: m(torch.randn(2, 128)))
    if grouped:  # one per expert, as Transformer Engine's grouped linear holds them
        model[0].weight_quantizer = GroupedQuantizer(model[0].weight_quantizer)

    with pytest.raises(NotImplementedError, match="W4A8"):
        mtq.compress(model)

    assert not is_real_quantized(model)


@pytest.mark.parametrize(
    "cast",
    [
        pytest.param(lambda m: m.to(dtype=torch.bfloat16), id="to_dtype"),
        pytest.param(lambda m: m.half().bfloat16(), id="half_then_bfloat16"),
    ],
)
def test_dtype_cast_keeps_real_quant_scales(cast):
    """Casting a compressed model keeps its scales, so the dequantized weights do not change."""
    model = nn.Sequential(nn.Linear(256, 256)).bfloat16()
    x = torch.randn(8, 256, dtype=torch.bfloat16)
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(x))
    mtq.compress(model)
    weight_quantizer = model[0].weight_quantizer
    scale = weight_quantizer._scale.clone()
    with torch.no_grad():
        expected = model(x)

    cast(model)

    assert weight_quantizer._scale.dtype == scale.dtype == torch.float32
    assert torch.equal(weight_quantizer._scale, scale)
    with torch.no_grad():
        assert torch.equal(model(x), expected)
