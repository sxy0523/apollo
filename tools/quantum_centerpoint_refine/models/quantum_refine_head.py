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

"""Pure NumPy QuantumRefineHead prototype.

This is intentionally an offline research implementation. It does not depend on
PennyLane, Qiskit, PyTorch, or Apollo runtime targets.
"""

import json
import os
from typing import Dict, Optional

import numpy as np


DELTA_DIM = 8


class Linear:
  """Small NumPy linear layer."""

  def __init__(self, weight: np.ndarray, bias: np.ndarray):
    self.weight = np.asarray(weight, dtype=np.float32)
    self.bias = np.asarray(bias, dtype=np.float32)

  @classmethod
  def create(cls, input_dim: int, output_dim: int, rng: np.random.RandomState):
    limit = np.sqrt(6.0 / float(input_dim + output_dim))
    weight = rng.uniform(-limit, limit, size=(input_dim, output_dim)).astype(
        np.float32
    )
    bias = np.zeros((output_dim,), dtype=np.float32)
    return cls(weight=weight, bias=bias)

  def __call__(self, values: np.ndarray) -> np.ndarray:
    return values @ self.weight + self.bias


def _ry(theta: float) -> np.ndarray:
  half = theta * 0.5
  return np.asarray(
      [[np.cos(half), -np.sin(half)], [np.sin(half), np.cos(half)]],
      dtype=np.complex64,
  )


def _apply_single_qubit_gate(
    state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
  next_state = np.zeros_like(state)
  for basis in range(1 << n_qubits):
    bit = (basis >> qubit) & 1
    partner = basis ^ (1 << qubit)
    if bit == 0:
      next_state[basis] += gate[0, 0] * state[basis]
      next_state[partner] += gate[1, 0] * state[basis]
    else:
      next_state[partner] += gate[0, 1] * state[basis]
      next_state[basis] += gate[1, 1] * state[basis]
  return next_state


def _apply_cnot(
    state: np.ndarray, control: int, target: int, n_qubits: int
) -> np.ndarray:
  next_state = np.zeros_like(state)
  for basis in range(1 << n_qubits):
    out_basis = basis
    if (basis >> control) & 1:
      out_basis = basis ^ (1 << target)
    next_state[out_basis] = state[basis]
  return next_state


def _pauli_z_expectations(state: np.ndarray, n_qubits: int) -> np.ndarray:
  probabilities = np.abs(state) ** 2
  values = np.zeros((n_qubits,), dtype=np.float32)
  for qubit in range(n_qubits):
    total = 0.0
    for basis, probability in enumerate(probabilities):
      sign = -1.0 if ((basis >> qubit) & 1) else 1.0
      total += sign * float(probability)
    values[qubit] = total
  return values


class QuantumRefineHead:
  """Offline proposal residual teacher with a small simulated quantum circuit."""

  def __init__(
      self,
      proposal_feature_dim: int,
      n_qubits: int = 4,
      n_layers: int = 2,
      seed: int = 2026,
  ):
    if proposal_feature_dim <= 0:
      raise ValueError("proposal_feature_dim must be positive")
    if n_qubits <= 0:
      raise ValueError("n_qubits must be positive")
    if n_layers <= 0:
      raise ValueError("n_layers must be positive")
    self.proposal_feature_dim = proposal_feature_dim
    self.n_qubits = n_qubits
    self.n_layers = n_layers
    self.rng = np.random.RandomState(seed)
    self.projector = Linear.create(proposal_feature_dim, n_qubits, self.rng)
    self.circuit_angles = self.rng.uniform(
        -0.05, 0.05, size=(n_layers, n_qubits)
    ).astype(np.float32)
    self.output_head = Linear.create(n_qubits, DELTA_DIM, self.rng)

  def _circuit(self, angles: np.ndarray) -> np.ndarray:
    state = np.zeros((1 << self.n_qubits,), dtype=np.complex64)
    state[0] = 1.0 + 0.0j

    for qubit in range(self.n_qubits):
      state = _apply_single_qubit_gate(
          state, _ry(float(angles[qubit])), qubit, self.n_qubits
      )

    for layer in range(self.n_layers):
      for qubit in range(self.n_qubits):
        state = _apply_single_qubit_gate(
            state,
            _ry(float(self.circuit_angles[layer, qubit])),
            qubit,
            self.n_qubits,
        )
      for qubit in range(self.n_qubits):
        state = _apply_cnot(
            state, qubit, (qubit + 1) % self.n_qubits, self.n_qubits
        )

    return _pauli_z_expectations(state, self.n_qubits)

  def quantum_features(self, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != self.proposal_feature_dim:
      raise ValueError(
          f"features must have shape [N, {self.proposal_feature_dim}], got "
          f"{features.shape}"
      )
    angles = self.projector(features)
    encoded = [self._circuit(row) for row in angles]
    return np.asarray(encoded, dtype=np.float32)

  def forward(self, features: np.ndarray) -> np.ndarray:
    return self.output_head(self.quantum_features(features)).astype(np.float32)

  __call__ = forward

  def fit_output_head(
      self, features: np.ndarray, targets: np.ndarray, ridge_l2: float = 1.0e-4
  ) -> None:
    """Fits the final linear head with ridge regression."""
    hidden = self.quantum_features(features)
    ones = np.ones((hidden.shape[0], 1), dtype=np.float32)
    design = np.concatenate([hidden, ones], axis=1)
    regularizer = ridge_l2 * np.eye(design.shape[1], dtype=np.float32)
    regularizer[-1, -1] = 0.0
    solution = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    self.output_head.weight = solution[:-1, :].astype(np.float32)
    self.output_head.bias = solution[-1, :].astype(np.float32)

  def state_dict(self) -> Dict[str, np.ndarray]:
    return {
        "proposal_feature_dim": np.asarray(self.proposal_feature_dim),
        "n_qubits": np.asarray(self.n_qubits),
        "n_layers": np.asarray(self.n_layers),
        "projector_weight": self.projector.weight,
        "projector_bias": self.projector.bias,
        "circuit_angles": self.circuit_angles,
        "output_weight": self.output_head.weight,
        "output_bias": self.output_head.bias,
    }

  def save_checkpoint(
      self, path: str, metadata: Optional[Dict[str, object]] = None) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    state = self.state_dict()
    state["metadata_json"] = np.asarray(json.dumps(metadata or {}, sort_keys=True))
    np.savez(path, **state)

  @classmethod
  def load_checkpoint(cls, path: str) -> "QuantumRefineHead":
    data = np.load(path, allow_pickle=False)
    model = cls(
        proposal_feature_dim=int(data["proposal_feature_dim"]),
        n_qubits=int(data["n_qubits"]),
        n_layers=int(data["n_layers"]),
    )
    model.projector.weight = data["projector_weight"].astype(np.float32)
    model.projector.bias = data["projector_bias"].astype(np.float32)
    model.circuit_angles = data["circuit_angles"].astype(np.float32)
    model.output_head.weight = data["output_weight"].astype(np.float32)
    model.output_head.bias = data["output_bias"].astype(np.float32)
    return model
