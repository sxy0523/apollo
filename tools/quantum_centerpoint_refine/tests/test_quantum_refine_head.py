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

"""Tests for QuantumRefineHead."""

import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from datasets.proposal_dataset import ProposalDataset  # pylint: disable=wrong-import-position
from models.losses import mse_loss  # pylint: disable=wrong-import-position
from models.quantum_refine_head import (  # pylint: disable=wrong-import-position
    QuantumRefineHead,
)


class QuantumRefineHeadTest(unittest.TestCase):

  def test_forward_shape_is_batch_by_eight(self):
    dataset = ProposalDataset.synthetic(num_samples=5, seed=11)
    features, _ = dataset.as_arrays()
    model = QuantumRefineHead(proposal_feature_dim=9, n_qubits=4, n_layers=2)

    output = model(features)

    self.assertEqual(output.shape, (5, 8))
    self.assertTrue(np.all(np.isfinite(output)))

  def test_quantum_feature_shape(self):
    dataset = ProposalDataset.synthetic(num_samples=3, seed=12)
    features, _ = dataset.as_arrays()
    model = QuantumRefineHead(proposal_feature_dim=9, n_qubits=4, n_layers=2)

    hidden = model.quantum_features(features)

    self.assertEqual(hidden.shape, (3, 4))
    self.assertTrue(np.all(hidden <= 1.0001))
    self.assertTrue(np.all(hidden >= -1.0001))

  def test_output_head_fit_reduces_loss(self):
    dataset = ProposalDataset.synthetic(num_samples=32, seed=13)
    features, targets = dataset.as_arrays()
    model = QuantumRefineHead(proposal_feature_dim=9, n_qubits=4, n_layers=2)
    before = mse_loss(model(features), targets)

    model.fit_output_head(features, targets, ridge_l2=1.0e-4)
    after = mse_loss(model(features), targets)

    self.assertLess(after, before)

  def test_checkpoint_round_trip(self):
    dataset = ProposalDataset.synthetic(num_samples=4, seed=14)
    features, _ = dataset.as_arrays()
    model = QuantumRefineHead(proposal_feature_dim=9, n_qubits=4, n_layers=2)
    expected = model(features)

    with tempfile.TemporaryDirectory() as tmp_dir:
      checkpoint = os.path.join(tmp_dir, "q_refine_checkpoint.npz")
      model.save_checkpoint(checkpoint, metadata={"test": True})
      restored = QuantumRefineHead.load_checkpoint(checkpoint)

    actual = restored(features)
    self.assertTrue(np.allclose(actual, expected))


if __name__ == "__main__":
  unittest.main()
