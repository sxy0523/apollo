#!/usr/bin/env python3
###############################################################################
# Copyright 2026 The Apollo Authors. All Rights Reserved.
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
###############################################################################

"""Minimal ONNX exporter for the NumPy MLP student.

This writes a standard ONNX ModelProto without importing the onnx package. The
graph is a feed-forward Gemm/Relu network whose input is already normalized
proposal features.
"""

import os

import numpy as np


ONNX_FLOAT = 1


def _varint(value):
  value = int(value)
  encoded = bytearray()
  while value > 0x7F:
    encoded.append((value & 0x7F) | 0x80)
    value >>= 7
  encoded.append(value)
  return bytes(encoded)


def _key(field_number, wire_type):
  return _varint((field_number << 3) | wire_type)


def _int_field(field_number, value):
  return _key(field_number, 0) + _varint(value)


def _string_field(field_number, value):
  data = value.encode("utf-8")
  return _key(field_number, 2) + _varint(len(data)) + data


def _bytes_field(field_number, value):
  return _key(field_number, 2) + _varint(len(value)) + value


def _message_field(field_number, payload):
  return _key(field_number, 2) + _varint(len(payload)) + payload


def _tensor(name, array):
  array = np.asarray(array, dtype=np.float32)
  payload = b""
  for dim in array.shape:
    payload += _int_field(1, int(dim))
  payload += _int_field(2, ONNX_FLOAT)
  payload += _string_field(8, name)
  payload += _bytes_field(9, np.ascontiguousarray(array, dtype="<f4").tobytes())
  return payload


def _dimension_value(value):
  return _int_field(1, value)


def _dimension_param(value):
  return _string_field(2, value)


def _value_info(name, second_dim):
  shape = _message_field(1, _dimension_param("batch"))
  shape += _message_field(1, _dimension_value(second_dim))
  tensor_type = _int_field(1, ONNX_FLOAT) + _message_field(2, shape)
  type_proto = _message_field(1, tensor_type)
  return _string_field(1, name) + _message_field(2, type_proto)


def _node(op_type, name, inputs, outputs):
  payload = b""
  for input_name in inputs:
    payload += _string_field(1, input_name)
  for output_name in outputs:
    payload += _string_field(2, output_name)
  payload += _string_field(3, name)
  payload += _string_field(4, op_type)
  return payload


def _opset_import(domain, version):
  return _string_field(1, domain) + _int_field(2, version)


def export_student_onnx(student, output_path):
  """Exports an MLPStudent to an ONNX Gemm/Relu graph."""
  graph = b""
  previous = "input"
  nodes = []
  initializers = []
  for index, (weight, bias) in enumerate(zip(student.weights, student.biases)):
    weight_name = f"weight_{index}"
    bias_name = f"bias_{index}"
    gemm_output = "output" if index == len(student.weights) - 1 else f"gemm_{index}"
    nodes.append(
        _node("Gemm", f"gemm_{index}", [previous, weight_name, bias_name],
              [gemm_output])
    )
    initializers.append(_tensor(weight_name, weight))
    initializers.append(_tensor(bias_name, bias))
    if index != len(student.weights) - 1:
      relu_output = f"relu_{index}"
      nodes.append(_node("Relu", f"relu_{index}", [gemm_output], [relu_output]))
      previous = relu_output

  for node in nodes:
    graph += _message_field(1, node)
  graph += _string_field(2, "quantum_centerpoint_mlp_student")
  for initializer in initializers:
    graph += _message_field(5, initializer)
  graph += _message_field(11, _value_info("input", student.input_dim))
  graph += _message_field(12, _value_info("output", student.output_dim))

  model = _int_field(1, 7)
  model += _string_field(2, "apollo_quantum_centerpoint_refine")
  model += _string_field(3, "phase7")
  model += _message_field(7, graph)
  model += _message_field(8, _opset_import("", 13))

  output_dir = os.path.dirname(output_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(output_path, "wb") as output_file:
    output_file.write(model)
  return output_path
