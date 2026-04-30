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

"""Tests for the distilled MLP student."""

import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from models.losses import mse_loss  # pylint: disable=wrong-import-position
from models.mlp_student import MLPStudent  # pylint: disable=wrong-import-position


class MLPStudentTest(unittest.TestCase):

  def test_student_forward_shape(self):
    student = MLPStudent(input_dim=9, seed=3)
    features = np.zeros((6, 9), dtype=np.float32)

    output = student(features)

    self.assertEqual(output.shape, (6, 8))

  def test_train_step_reduces_distill_loss(self):
    rng = np.random.RandomState(4)
    features = rng.normal(size=(32, 9)).astype(np.float32)
    target = np.zeros((32, 8), dtype=np.float32)
    student = MLPStudent(input_dim=9, seed=4)
    before = mse_loss(student(features), target)

    for _ in range(40):
      student.train_step(features, target, learning_rate=2.0e-2)
    after = mse_loss(student(features), target)

    self.assertLess(after, before)

  def test_checkpoint_round_trip(self):
    rng = np.random.RandomState(5)
    features = rng.normal(size=(4, 9)).astype(np.float32)
    student = MLPStudent(input_dim=9, seed=5)
    expected = student(features)

    with tempfile.TemporaryDirectory() as tmp_dir:
      checkpoint = os.path.join(tmp_dir, "student_checkpoint.npz")
      student.save_checkpoint(checkpoint, metadata_json='{"ok": true}')
      restored = MLPStudent.load_checkpoint(checkpoint)

    self.assertTrue(np.allclose(restored(features), expected))


if __name__ == "__main__":
  unittest.main()
