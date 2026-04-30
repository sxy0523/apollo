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

"""Evaluates an offline QuantumRefineHead checkpoint."""

import argparse
import json
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from datasets.proposal_dataset import ProposalDataset  # pylint: disable=wrong-import-position
from models.losses import mae, mse_loss  # pylint: disable=wrong-import-position
from models.quantum_refine_head import (  # pylint: disable=wrong-import-position
    QuantumRefineHead,
)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--export-path", default="")
  parser.add_argument("--target-path", default="")
  parser.add_argument("--synthetic", action="store_true")
  parser.add_argument("--num-synthetic", type=int, default=64)
  parser.add_argument("--seed", type=int, default=2026)
  args = parser.parse_args()

  if args.synthetic or not args.export_path:
    dataset = ProposalDataset.synthetic(args.num_synthetic, seed=args.seed)
  else:
    dataset = ProposalDataset.from_export_path(
        args.export_path,
        target_path=args.target_path,
        use_synthetic_targets=not args.target_path,
    )

  model = QuantumRefineHead.load_checkpoint(args.checkpoint)
  features, targets = dataset.as_arrays()
  prediction = model(features)
  result = {
      "num_samples": int(features.shape[0]),
      "mse": mse_loss(prediction, targets),
      "mae": mae(prediction, targets),
      "output_shape": list(prediction.shape),
  }
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
