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

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.quantization.graph_rewrites import remove_redundant_cast_nodes


def _shared_initializer_model():
    # W (fp32) -> Cast(fp16) -> Mul ; W -> Add (fp32)
    nodes = [
        helper.make_node("Cast", ["W"], ["W_fp16"], to=TensorProto.FLOAT16),
        helper.make_node("Mul", ["x16", "W_fp16"], ["y16"]),
        helper.make_node("Add", ["x32", "W"], ["y32"]),
    ]
    inputs = [
        helper.make_tensor_value_info("x16", TensorProto.FLOAT16, [2, 3]),
        helper.make_tensor_value_info("x32", TensorProto.FLOAT, [2, 3]),
    ]
    outputs = [
        helper.make_tensor_value_info("y16", TensorProto.FLOAT16, [2, 3]),
        helper.make_tensor_value_info("y32", TensorProto.FLOAT, [2, 3]),
    ]
    return nodes, inputs, outputs


def _two_casts_model():
    # W (fp32) -> Cast(fp16) -> Mul ; W -> Cast(fp64) -> Add
    nodes = [
        helper.make_node("Cast", ["W"], ["W_fp16"], to=TensorProto.FLOAT16),
        helper.make_node("Mul", ["x16", "W_fp16"], ["y16"]),
        helper.make_node("Cast", ["W"], ["W_fp64"], to=TensorProto.DOUBLE),
        helper.make_node("Add", ["x64", "W_fp64"], ["y64"]),
    ]
    inputs = [
        helper.make_tensor_value_info("x16", TensorProto.FLOAT16, [2, 3]),
        helper.make_tensor_value_info("x64", TensorProto.DOUBLE, [2, 3]),
    ]
    outputs = [
        helper.make_tensor_value_info("y16", TensorProto.FLOAT16, [2, 3]),
        helper.make_tensor_value_info("y64", TensorProto.DOUBLE, [2, 3]),
    ]
    return nodes, inputs, outputs


def _run(model, feeds):
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


@pytest.mark.parametrize("build", [_shared_initializer_model, _two_casts_model])
def test_remove_redundant_cast_keeps_shared_initializer_valid(build):
    nodes, inputs, outputs = build()
    weight = np.array([[1.1, -2.2, 3.3], [0.1, 1e-4, 7.0]], dtype=np.float32)
    graph = helper.make_graph(
        nodes, "g", inputs, outputs, initializer=[numpy_helper.from_array(weight, "W")]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model, full_check=True)

    rng = np.random.default_rng(0)
    feeds = {
        vi.name: rng.standard_normal((2, 3)).astype(
            helper.tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type)
        )
        for vi in inputs
    }
    expected = _run(model, feeds)

    remove_redundant_cast_nodes(model.graph)

    assert not [n for n in model.graph.node if n.op_type == "Cast"]
    onnx.checker.check_model(model, full_check=True)
    for got, want in zip(_run(model, feeds), expected):
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(got, want)
