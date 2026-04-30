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

"""Tests for proposal box delta helpers."""

import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from models.losses import apply_box_delta  # pylint: disable=wrong-import-position


class BoxDeltaTest(unittest.TestCase):

  def test_apply_box_delta_preserves_class_and_clamps(self):
    features = np.asarray(
        [[1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 3.13, 0.98, 4.0]],
        dtype=np.float32,
    )
    deltas = np.asarray(
        [[0.5, -0.5, 0.25, -2.0, -2.0, -2.0, 0.2, 0.5]],
        dtype=np.float32,
    )

    refined = apply_box_delta(features, deltas)

    self.assertAlmostEqual(float(refined[0, 0]), 1.5)
    self.assertGreaterEqual(float(refined[0, 3]), 1.0e-3)
    self.assertGreaterEqual(float(refined[0, 4]), 1.0e-3)
    self.assertGreaterEqual(float(refined[0, 5]), 1.0e-3)
    self.assertLessEqual(float(refined[0, 6]), np.pi)
    self.assertGreaterEqual(float(refined[0, 6]), -np.pi)
    self.assertAlmostEqual(float(refined[0, 7]), 1.0)
    self.assertAlmostEqual(float(refined[0, 8]), 4.0)

  def test_shape_mismatch_raises(self):
    with self.assertRaises(ValueError):
      apply_box_delta(np.zeros((1, 9), dtype=np.float32),
                      np.zeros((2, 8), dtype=np.float32))


if __name__ == "__main__":
  unittest.main()
