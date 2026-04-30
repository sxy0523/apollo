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

"""Tests for ONNX export."""

import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from models.mlp_student import MLPStudent  # pylint: disable=wrong-import-position
from models.onnx_export import export_student_onnx  # pylint: disable=wrong-import-position


def _read_varint(data, offset):
  shift = 0
  value = 0
  while True:
    byte = data[offset]
    offset += 1
    value |= (byte & 0x7F) << shift
    if not byte & 0x80:
      return value, offset
    shift += 7


def _iter_fields(data):
  offset = 0
  while offset < len(data):
    key, offset = _read_varint(data, offset)
    field_number = key >> 3
    wire_type = key & 0x07
    if wire_type == 0:
      value, offset = _read_varint(data, offset)
      yield field_number, wire_type, value
    elif wire_type == 2:
      length, offset = _read_varint(data, offset)
      value = data[offset:offset + length]
      offset += length
      yield field_number, wire_type, value
    else:
      raise ValueError("unsupported protobuf wire type")


def _parse_tensor(data):
  dims = []
  name = ""
  raw_data = b""
  for field_number, _, value in _iter_fields(data):
    if field_number == 1:
      dims.append(int(value))
    elif field_number == 8:
      name = value.decode("utf-8")
    elif field_number == 9:
      raw_data = value
  return name, np.frombuffer(raw_data, dtype="<f4").reshape(dims)


def _read_initializers(onnx_path):
  with open(onnx_path, "rb") as model_file:
    model = model_file.read()
  graph = None
  for field_number, _, value in _iter_fields(model):
    if field_number == 7:
      graph = value
      break
  if graph is None:
    raise ValueError("ONNX graph field not found")

  initializers = {}
  for field_number, _, value in _iter_fields(graph):
    if field_number == 5:
      name, tensor = _parse_tensor(value)
      initializers[name] = tensor
  return initializers


def _run_exported_mlp(onnx_path, features):
  initializers = _read_initializers(onnx_path)
  values = features
  layer = 0
  while f"weight_{layer}" in initializers:
    values = values @ initializers[f"weight_{layer}"] + initializers[
        f"bias_{layer}"]
    if f"weight_{layer + 1}" in initializers:
      values = np.maximum(values, 0.0)
    layer += 1
  return values.astype(np.float32)


class ONNXExportTest(unittest.TestCase):

  def test_export_student_onnx(self):
    student = MLPStudent(input_dim=9, seed=6)
    with tempfile.TemporaryDirectory() as tmp_dir:
      output_path = os.path.join(tmp_dir, "student_model.onnx")

      export_student_onnx(student, output_path)

      self.assertTrue(os.path.exists(output_path))
      self.assertGreater(os.path.getsize(output_path), 0)

  def test_exported_onnx_weights_match_numpy(self):
    student = MLPStudent(input_dim=9, seed=8)
    features = np.random.RandomState(8).normal(size=(3, 9)).astype(np.float32)
    expected = student(features)
    with tempfile.TemporaryDirectory() as tmp_dir:
      output_path = os.path.join(tmp_dir, "student_model.onnx")
      export_student_onnx(student, output_path)

      actual = _run_exported_mlp(output_path, features)

    self.assertTrue(np.allclose(actual, expected, atol=1.0e-6))

  def test_onnxruntime_matches_numpy_if_available(self):
    try:
      import onnxruntime as ort  # pylint: disable=import-error
    except ImportError:
      self.skipTest("onnxruntime is not installed")

    student = MLPStudent(input_dim=9, seed=7)
    features = np.random.RandomState(7).normal(size=(3, 9)).astype(np.float32)
    expected = student(features)
    with tempfile.TemporaryDirectory() as tmp_dir:
      output_path = os.path.join(tmp_dir, "student_model.onnx")
      export_student_onnx(student, output_path)
      try:
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
      except TypeError:
        session = ort.InferenceSession(output_path)
      actual = session.run(None, {"input": features})[0]

    self.assertTrue(np.allclose(actual, expected, atol=1.0e-5))


if __name__ == "__main__":
  unittest.main()
