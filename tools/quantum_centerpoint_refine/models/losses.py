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

"""Losses and box delta helpers for offline proposal refinement."""

import numpy as np


def mse_loss(prediction: np.ndarray, target: np.ndarray) -> float:
  diff = np.asarray(prediction, dtype=np.float32) - np.asarray(target, dtype=np.float32)
  return float(np.mean(diff * diff))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
  diff = np.asarray(prediction, dtype=np.float32) - np.asarray(target, dtype=np.float32)
  return float(np.mean(np.abs(diff)))


def normalize_angle(angle: np.ndarray) -> np.ndarray:
  return (angle + np.pi) % (2.0 * np.pi) - np.pi


def apply_box_delta(
    features: np.ndarray,
    deltas: np.ndarray,
    dimension_min: float = 1.0e-3,
) -> np.ndarray:
  """Applies 8 residual deltas to 9 proposal features.

  The class id column is copied unchanged.
  """
  features = np.asarray(features, dtype=np.float32)
  deltas = np.asarray(deltas, dtype=np.float32)
  if features.ndim != 2 or features.shape[1] != 9:
    raise ValueError("features must have shape [N, 9]")
  if deltas.ndim != 2 or deltas.shape[1] != 8:
    raise ValueError("deltas must have shape [N, 8]")
  if features.shape[0] != deltas.shape[0]:
    raise ValueError("features and deltas must have matching row counts")

  refined = features.copy()
  refined[:, 0:7] += deltas[:, 0:7]
  refined[:, 3:6] = np.maximum(refined[:, 3:6], dimension_min)
  refined[:, 6] = normalize_angle(refined[:, 6])
  refined[:, 7] = np.clip(refined[:, 7] + deltas[:, 7], 0.0, 1.0)
  return refined.astype(np.float32)
