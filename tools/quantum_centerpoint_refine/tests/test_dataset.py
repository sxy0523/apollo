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

"""Tests for proposal dataset loading."""

import os
import struct
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from datasets.proposal_dataset import (  # pylint: disable=wrong-import-position
    DELTA_ORDER,
    FEATURE_ORDER,
    HEADER,
    RECORD,
    ProposalDataset,
    read_proposal_export,
)


class ProposalDatasetTest(unittest.TestCase):

  def _write_export(self, path):
    with open(path, "wb") as writer:
      writer.write(HEADER.pack(b"APCPROP1", 1, RECORD.size, 3, 12.5, 2, 9))
      writer.write(RECORD.pack(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.9, 2))
      writer.write(RECORD.pack(7.0, 8.0, 9.0, 4.5, 1.5, 1.8, -0.2, 0.7, 1))

  def test_read_proposal_export(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      path = os.path.join(tmp_dir, "proposal_000000.bin")
      self._write_export(path)

      features = read_proposal_export(path)

    self.assertEqual(features.shape, (2, len(FEATURE_ORDER)))
    self.assertAlmostEqual(float(features[0, 0]), 1.0)
    self.assertAlmostEqual(float(features[1, 7]), 0.7, places=5)

  def test_dataset_from_export_uses_synthetic_targets(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      self._write_export(os.path.join(tmp_dir, "proposal_000000.bin"))

      dataset = ProposalDataset.from_export_path(tmp_dir)

    self.assertEqual(dataset.features.shape, (2, len(FEATURE_ORDER)))
    self.assertEqual(dataset.targets.shape, (2, len(DELTA_ORDER)))
    self.assertTrue(np.all(np.isfinite(dataset.targets)))

  def test_synthetic_split(self):
    dataset = ProposalDataset.synthetic(num_samples=10, seed=7)

    train_dataset, val_dataset = dataset.split(validation_fraction=0.3, seed=7)

    self.assertEqual(train_dataset.features.shape[0], 7)
    self.assertEqual(val_dataset.features.shape[0], 3)


if __name__ == "__main__":
  unittest.main()
