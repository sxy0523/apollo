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

"""Proposal export dataset utilities for offline refine training."""

import csv
import glob
import os
import struct
from typing import List, Optional, Tuple

import numpy as np


FEATURE_ORDER = (
    "x",
    "y",
    "z",
    "length",
    "width",
    "height",
    "yaw",
    "score",
    "class_id",
)
DELTA_ORDER = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_length",
    "delta_width",
    "delta_height",
    "delta_yaw",
    "delta_score",
)

HEADER = struct.Struct("<8sIIQdII")
RECORD = struct.Struct("<ffffffffi")


def _read_exact(reader, num_bytes: int) -> bytes:
  data = reader.read(num_bytes)
  if len(data) != num_bytes:
    raise ValueError("unexpected end of proposal export file")
  return data


def read_proposal_export(path: str) -> np.ndarray:
  """Reads a Phase 5 proposal export file into a float32 feature matrix."""
  with open(path, "rb") as reader:
    magic, version, record_size, _, _, count, feature_dim = HEADER.unpack(
        _read_exact(reader, HEADER.size)
    )
    if magic != b"APCPROP1":
      raise ValueError(f"{path}: unexpected proposal export magic")
    if version != 1 or record_size != RECORD.size:
      raise ValueError(f"{path}: unsupported proposal export version")
    if feature_dim != len(FEATURE_ORDER):
      raise ValueError(f"{path}: unexpected feature_dim {feature_dim}")

    features = np.zeros((count, len(FEATURE_ORDER)), dtype=np.float32)
    for row in range(count):
      values = RECORD.unpack(_read_exact(reader, RECORD.size))
      features[row, :] = np.asarray(values, dtype=np.float32)
  return features


def _proposal_files(path: str) -> List[str]:
  if os.path.isfile(path):
    return [path]
  return sorted(glob.glob(os.path.join(path, "proposal_*.bin")))


def load_proposal_exports(path: str) -> np.ndarray:
  """Loads one proposal export file or all export files in a directory."""
  files = _proposal_files(path)
  if not files:
    raise FileNotFoundError(f"no proposal_*.bin files found under {path}")
  matrices = [read_proposal_export(file_path) for file_path in files]
  if not matrices:
    return np.zeros((0, len(FEATURE_ORDER)), dtype=np.float32)
  return np.concatenate(matrices, axis=0)


def load_residual_targets(path: str, expected_rows: Optional[int] = None) -> np.ndarray:
  """Loads supervised residual targets from .npy or CSV."""
  if path.endswith(".npy"):
    targets = np.load(path).astype(np.float32)
  else:
    rows = []
    with open(path, newline="") as csv_file:
      reader = csv.reader(csv_file)
      for row in reader:
        if not row:
          continue
        try:
          rows.append([float(value) for value in row[:len(DELTA_ORDER)]])
        except ValueError:
          continue
    targets = np.asarray(rows, dtype=np.float32)

  if targets.ndim != 2 or targets.shape[1] != len(DELTA_ORDER):
    raise ValueError(
        f"target matrix must have shape [N, {len(DELTA_ORDER)}], got "
        f"{targets.shape}"
    )
  if expected_rows is not None and targets.shape[0] != expected_rows:
    raise ValueError(
        f"target row count {targets.shape[0]} does not match proposals "
        f"{expected_rows}"
    )
  return targets


def make_synthetic_targets(features: np.ndarray) -> np.ndarray:
  """Creates deterministic pseudo residual targets for pipeline testing."""
  if features.ndim != 2 or features.shape[1] != len(FEATURE_ORDER):
    raise ValueError("features must have shape [N, 9]")

  x = features[:, 0]
  y = features[:, 1]
  z = features[:, 2]
  length = np.maximum(features[:, 3], 1.0e-3)
  width = np.maximum(features[:, 4], 1.0e-3)
  height = np.maximum(features[:, 5], 1.0e-3)
  yaw = features[:, 6]
  score = np.clip(features[:, 7], 0.0, 1.0)

  targets = np.zeros((features.shape[0], len(DELTA_ORDER)), dtype=np.float32)
  targets[:, 0] = 0.02 * np.tanh(y / 20.0)
  targets[:, 1] = -0.02 * np.tanh(x / 20.0)
  targets[:, 2] = 0.01 * np.tanh(z / 4.0)
  targets[:, 3] = 0.01 * np.tanh((length - 4.0) / 4.0)
  targets[:, 4] = 0.01 * np.tanh((width - 1.8) / 2.0)
  targets[:, 5] = 0.01 * np.tanh((height - 1.6) / 2.0)
  targets[:, 6] = 0.03 * np.sin(yaw)
  targets[:, 7] = 0.05 * (1.0 - score)
  return targets


def make_synthetic_features(num_samples: int, seed: int = 2026) -> np.ndarray:
  """Creates deterministic proposal-like features for smoke tests."""
  rng = np.random.RandomState(seed)
  features = np.zeros((num_samples, len(FEATURE_ORDER)), dtype=np.float32)
  features[:, 0] = rng.uniform(-50.0, 50.0, size=num_samples)
  features[:, 1] = rng.uniform(-30.0, 30.0, size=num_samples)
  features[:, 2] = rng.uniform(-2.5, 3.0, size=num_samples)
  features[:, 3] = rng.uniform(2.0, 6.0, size=num_samples)
  features[:, 4] = rng.uniform(0.8, 2.5, size=num_samples)
  features[:, 5] = rng.uniform(1.0, 3.0, size=num_samples)
  features[:, 6] = rng.uniform(-np.pi, np.pi, size=num_samples)
  features[:, 7] = rng.uniform(0.05, 0.99, size=num_samples)
  features[:, 8] = rng.randint(0, 5, size=num_samples)
  return features


class ProposalDataset:
  """In-memory proposal dataset used by the offline scripts."""

  def __init__(self, features: np.ndarray, targets: np.ndarray):
    self.features = np.asarray(features, dtype=np.float32)
    self.targets = np.asarray(targets, dtype=np.float32)

  @classmethod
  def from_export_path(
      cls,
      export_path: str,
      target_path: Optional[str] = None,
      use_synthetic_targets: bool = True,
  ) -> "ProposalDataset":
    features = load_proposal_exports(export_path)
    if target_path:
      targets = load_residual_targets(target_path, expected_rows=features.shape[0])
    elif use_synthetic_targets:
      targets = make_synthetic_targets(features)
    else:
      targets = np.zeros((features.shape[0], len(DELTA_ORDER)), dtype=np.float32)
    return cls(features=features, targets=targets)

  @classmethod
  def synthetic(cls, num_samples: int, seed: int = 2026) -> "ProposalDataset":
    features = make_synthetic_features(num_samples, seed=seed)
    return cls(features=features, targets=make_synthetic_targets(features))

  def split(
      self, validation_fraction: float = 0.2, seed: int = 2026
  ) -> Tuple["ProposalDataset", "ProposalDataset"]:
    if not 0.0 <= validation_fraction < 1.0:
      raise ValueError("validation_fraction must be in [0, 1)")
    num_rows = self.features.shape[0]
    rng = np.random.RandomState(seed)
    order = rng.permutation(num_rows)
    num_val = int(round(num_rows * validation_fraction))
    val_idx = order[:num_val]
    train_idx = order[num_val:]
    return (
        ProposalDataset(self.features[train_idx], self.targets[train_idx]),
        ProposalDataset(self.features[val_idx], self.targets[val_idx]),
    )

  def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
    return self.features.astype(np.float32), self.targets.astype(np.float32)
