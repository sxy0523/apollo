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

"""Trains the offline NumPy QuantumRefineHead prototype."""

import argparse
import json
import os
import sys

import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from datasets.proposal_dataset import (  # pylint: disable=wrong-import-position
    DELTA_ORDER,
    FEATURE_ORDER,
    ProposalDataset,
)
from models.losses import mae, mse_loss  # pylint: disable=wrong-import-position
from models.quantum_refine_head import (  # pylint: disable=wrong-import-position
    QuantumRefineHead,
)


def _default_config_path():
  return os.path.join(ROOT, "configs", "q_refine.yaml")


def _load_config(path):
  with open(path) as config_file:
    return yaml.safe_load(config_file) or {}


def _dataset_from_args(args, config):
  seed = args.seed if args.seed is not None else int(config.get("seed", 2026))
  if args.synthetic or not args.export_path:
    synthetic_config = config.get("synthetic", {})
    num_samples = args.num_synthetic or int(synthetic_config.get("num_samples", 128))
    return ProposalDataset.synthetic(num_samples=num_samples, seed=seed)
  return ProposalDataset.from_export_path(
      args.export_path,
      target_path=args.target_path,
      use_synthetic_targets=not args.target_path,
  )


def _loss_summary(model, dataset):
  features, targets = dataset.as_arrays()
  if features.shape[0] == 0:
    return {"mse": None, "mae": None, "num_samples": 0}
  prediction = model(features)
  return {
      "mse": mse_loss(prediction, targets),
      "mae": mae(prediction, targets),
      "num_samples": int(features.shape[0]),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=_default_config_path())
  parser.add_argument("--export-path", default="")
  parser.add_argument("--target-path", default="")
  parser.add_argument("--output-dir", default="")
  parser.add_argument("--synthetic", action="store_true")
  parser.add_argument("--num-synthetic", type=int, default=0)
  parser.add_argument("--epochs", type=int, default=0)
  parser.add_argument("--seed", type=int, default=None)
  args = parser.parse_args()

  config = _load_config(args.config)
  seed = args.seed if args.seed is not None else int(config.get("seed", 2026))
  training_config = config.get("training", {})
  validation_fraction = float(config.get("validation_fraction", 0.2))
  epochs = args.epochs or int(training_config.get("epochs", 5))
  ridge_l2 = float(training_config.get("ridge_l2", 1.0e-4))
  output_dir = args.output_dir or training_config.get(
      "output_dir", "/tmp/quantum_centerpoint_refine"
  )

  dataset = _dataset_from_args(args, config)
  train_dataset, val_dataset = dataset.split(
      validation_fraction=validation_fraction, seed=seed
  )
  train_features, train_targets = train_dataset.as_arrays()

  model = QuantumRefineHead(
      proposal_feature_dim=int(config.get("proposal_feature_dim", len(FEATURE_ORDER))),
      n_qubits=int(config.get("n_qubits", 4)),
      n_layers=int(config.get("n_layers", 2)),
      seed=seed,
  )

  history = []
  history.append(
      {
          "epoch": 0,
          "train": _loss_summary(model, train_dataset),
          "val": _loss_summary(model, val_dataset),
      }
  )
  model.fit_output_head(train_features, train_targets, ridge_l2=ridge_l2)
  for epoch in range(1, max(1, epochs) + 1):
    history.append(
        {
            "epoch": epoch,
            "train": _loss_summary(model, train_dataset),
            "val": _loss_summary(model, val_dataset),
        }
    )

  os.makedirs(output_dir, exist_ok=True)
  checkpoint_path = os.path.join(output_dir, "q_refine_checkpoint.npz")
  log_path = os.path.join(output_dir, "train_log.json")
  metadata = {
      "feature_order": FEATURE_ORDER,
      "delta_order": DELTA_ORDER,
      "seed": seed,
      "n_qubits": model.n_qubits,
      "n_layers": model.n_layers,
      "training_method": "ridge_fit_output_head",
  }
  model.save_checkpoint(checkpoint_path, metadata=metadata)
  with open(log_path, "w") as log_file:
    json.dump({"history": history, "metadata": metadata}, log_file, indent=2)

  result = {
      "checkpoint": checkpoint_path,
      "log": log_path,
      "final": history[-1],
  }
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
