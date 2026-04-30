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

"""Small NumPy MLP student for distilling the offline quantum teacher."""

from typing import Dict, Iterable

import numpy as np


class MLPStudent:
  """input_dim -> 64 -> 64 -> 8 MLP used by the distillation phase."""

  def __init__(
      self,
      input_dim: int,
      hidden_dims: Iterable[int] = (64, 64),
      seed: int = 2026):
    self.input_dim = input_dim
    dims = [input_dim] + list(hidden_dims) + [8]
    rng = np.random.RandomState(seed)
    self.weights = []
    self.biases = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
      limit = np.sqrt(6.0 / float(in_dim + out_dim))
      self.weights.append(
          rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)
      )
      self.biases.append(np.zeros((out_dim,), dtype=np.float32))

  @property
  def output_dim(self) -> int:
    return int(self.biases[-1].shape[0])

  def forward(self, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
      values = values @ weight + bias
      if index != len(self.weights) - 1:
        values = np.maximum(values, 0.0)
    return values.astype(np.float32)

  __call__ = forward

  def _forward_with_cache(self, features: np.ndarray):
    activations = [np.asarray(features, dtype=np.float32)]
    pre_activations = []
    values = activations[0]
    for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
      pre_activation = values @ weight + bias
      pre_activations.append(pre_activation)
      if index != len(self.weights) - 1:
        values = np.maximum(pre_activation, 0.0)
      else:
        values = pre_activation
      activations.append(values)
    return values.astype(np.float32), activations, pre_activations

  def train_step(
      self,
      features: np.ndarray,
      teacher_target: np.ndarray,
      learning_rate: float = 1.0e-2,
      supervised_target: np.ndarray = None,
      lambda_sup: float = 0.0) -> float:
    """Runs one full-batch gradient step."""
    prediction, activations, pre_activations = self._forward_with_cache(features)
    target = np.asarray(teacher_target, dtype=np.float32)
    if supervised_target is not None and lambda_sup > 0.0:
      supervised_target = np.asarray(supervised_target, dtype=np.float32)
      grad = (prediction - target) + lambda_sup * (
          prediction - supervised_target)
      loss = np.mean((prediction - target) ** 2) + lambda_sup * np.mean(
          (prediction - supervised_target) ** 2)
    else:
      grad = prediction - target
      loss = np.mean((prediction - target) ** 2)

    grad *= 2.0 / float(np.prod(prediction.shape))
    weight_grads = []
    bias_grads = []
    for index in reversed(range(len(self.weights))):
      weight_grads.append(activations[index].T @ grad)
      bias_grads.append(np.sum(grad, axis=0))
      if index > 0:
        grad = grad @ self.weights[index].T
        grad = grad * (pre_activations[index - 1] > 0.0)

    for index in range(len(self.weights)):
      self.weights[index] -= learning_rate * weight_grads[-1 - index]
      self.biases[index] -= learning_rate * bias_grads[-1 - index]
    return float(loss)

  def state_dict(self) -> Dict[str, np.ndarray]:
    state = {"input_dim": np.asarray(self.input_dim)}
    for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
      state[f"weight_{index}"] = weight
      state[f"bias_{index}"] = bias
    return state

  def save_checkpoint(self, path: str, metadata_json: str = "") -> None:
    state = self.state_dict()
    state["metadata_json"] = np.asarray(metadata_json)
    np.savez(path, **state)

  @classmethod
  def load_checkpoint(cls, path: str) -> "MLPStudent":
    data = np.load(path, allow_pickle=False)
    num_layers = len([key for key in data.files if key.startswith("weight_")])
    hidden_dims = []
    for index in range(num_layers - 1):
      hidden_dims.append(int(data[f"weight_{index}"].shape[1]))
    student = cls(
        input_dim=int(data["input_dim"]),
        hidden_dims=hidden_dims,
        seed=0)
    for index in range(num_layers):
      student.weights[index] = data[f"weight_{index}"].astype(np.float32)
      student.biases[index] = data[f"bias_{index}"].astype(np.float32)
    return student
