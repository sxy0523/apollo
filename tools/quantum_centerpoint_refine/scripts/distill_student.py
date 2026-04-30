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

"""Distills an offline QuantumRefineHead teacher into an MLP student."""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from datasets.proposal_dataset import (  # pylint: disable=wrong-import-position
    DELTA_ORDER,
    FEATURE_ORDER,
    ProposalDataset,
)
from models.losses import mae, mse_loss  # pylint: disable=wrong-import-position
from models.mlp_student import MLPStudent  # pylint: disable=wrong-import-position
from models.onnx_export import export_student_onnx  # pylint: disable=wrong-import-position
from models.quantum_refine_head import (  # pylint: disable=wrong-import-position
    QuantumRefineHead,
)


def _git_hash():
  try:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=os.path.abspath(os.path.join(ROOT, "..", "..")),
        stderr=subprocess.DEVNULL).decode("utf-8").strip()
  except Exception:
    return ""


def _normalize(features):
  mean = np.mean(features, axis=0).astype(np.float32)
  std = np.std(features, axis=0).astype(np.float32)
  std = np.maximum(std, 1.0e-6)
  return ((features - mean) / std).astype(np.float32), mean, std


def _apply_normalization(features, mean, std):
  return ((features - mean) / std).astype(np.float32)


def _time_forward(fn, features, repeats):
  repeats = max(1, int(repeats))
  fn(features)
  start = time.perf_counter()
  for _ in range(repeats):
    fn(features)
  elapsed_ms = (time.perf_counter() - start) * 1000.0 / float(repeats)
  return elapsed_ms


def _dataset_from_args(args):
  if args.synthetic or not args.export_path:
    return ProposalDataset.synthetic(args.num_synthetic, seed=args.seed)
  return ProposalDataset.from_export_path(
      args.export_path,
      target_path=args.target_path,
      use_synthetic_targets=not args.target_path,
  )


def _metrics(student_prediction, teacher_output, supervised_target=None):
  result = {
      "teacher_student_mse": mse_loss(student_prediction, teacher_output),
      "delta_x_mae": mae(student_prediction[:, 0], teacher_output[:, 0]),
      "delta_y_mae": mae(student_prediction[:, 1], teacher_output[:, 1]),
      "delta_yaw_mae": mae(student_prediction[:, 6], teacher_output[:, 6]),
      "score_delta_mae": mae(student_prediction[:, 7], teacher_output[:, 7]),
  }
  if supervised_target is not None:
    result["supervised_mse"] = mse_loss(student_prediction, supervised_target)
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--teacher-checkpoint", required=True)
  parser.add_argument("--export-path", default="")
  parser.add_argument("--target-path", default="")
  parser.add_argument("--output-dir", default="/tmp/quantum_centerpoint_distill")
  parser.add_argument("--synthetic", action="store_true")
  parser.add_argument("--num-synthetic", type=int, default=128)
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--epochs", type=int, default=200)
  parser.add_argument("--learning-rate", type=float, default=1.0e-2)
  parser.add_argument("--lambda-sup", type=float, default=0.0)
  parser.add_argument("--latency-repeats", type=int, default=5)
  args = parser.parse_args()

  np.random.seed(args.seed)
  teacher = QuantumRefineHead.load_checkpoint(args.teacher_checkpoint)
  dataset = _dataset_from_args(args)
  features, supervised_targets = dataset.as_arrays()
  normalized_features, mean, std = _normalize(features)
  teacher_output = teacher(features)

  student = MLPStudent(input_dim=features.shape[1], seed=args.seed)
  history = []
  for epoch in range(args.epochs + 1):
    student_prediction = student(normalized_features)
    epoch_metrics = _metrics(
        student_prediction,
        teacher_output,
        supervised_targets if args.lambda_sup > 0.0 else None)
    epoch_metrics["epoch"] = epoch
    history.append(epoch_metrics)
    if epoch != args.epochs:
      student.train_step(
          normalized_features,
          teacher_output,
          learning_rate=args.learning_rate,
          supervised_target=supervised_targets,
          lambda_sup=args.lambda_sup)

  final_prediction = student(normalized_features)
  teacher_latency_ms = _time_forward(
      teacher, features, repeats=args.latency_repeats)
  student_latency_ms = _time_forward(
      student, normalized_features, repeats=args.latency_repeats * 20)
  final_metrics = _metrics(
      final_prediction,
      teacher_output,
      supervised_targets if args.lambda_sup > 0.0 else None)
  final_metrics["teacher_latency_ms"] = teacher_latency_ms
  final_metrics["student_latency_ms"] = student_latency_ms
  final_metrics["student_faster_than_teacher"] = (
      student_latency_ms < teacher_latency_ms)

  os.makedirs(args.output_dir, exist_ok=True)
  checkpoint_path = os.path.join(args.output_dir, "student_checkpoint.npz")
  onnx_path = os.path.join(args.output_dir, "student_model.onnx")
  metadata_path = os.path.join(args.output_dir, "student_metadata.json")
  log_path = os.path.join(args.output_dir, "distill_log.json")

  metadata = {
      "version": "phase7_numpy_mlp_student_v1",
      "input_dim": int(features.shape[1]),
      "output_dim": 8,
      "hidden_dims": [64, 64],
      "feature_order": list(FEATURE_ORDER),
      "delta_order": list(DELTA_ORDER),
      "normalization": {
          "mean": mean.astype(float).tolist(),
          "std": std.astype(float).tolist(),
      },
      "onnx_input": "normalized_proposal_features",
      "expected_refine_config": {
          "enable_proposal_refine": True,
          "proposal_refine_mode": "student_model",
          "proposal_refine_top_k": -1,
          "proposal_refine_score_threshold": 0.0,
          "proposal_refine_delta_scale": 1.0,
          "strict_refine": False,
      },
      "training_commit_hash": _git_hash(),
      "metrics": final_metrics,
  }

  student.save_checkpoint(checkpoint_path, metadata_json=json.dumps(metadata))
  export_student_onnx(student, onnx_path)
  with open(metadata_path, "w") as metadata_file:
    json.dump(metadata, metadata_file, indent=2, sort_keys=True)
  with open(log_path, "w") as log_file:
    json.dump({"history": history, "final": final_metrics}, log_file, indent=2)

  print(
      json.dumps(
          {
              "checkpoint": checkpoint_path,
              "onnx": onnx_path,
              "metadata": metadata_path,
              "log": log_path,
              "final": final_metrics,
          },
          indent=2,
          sort_keys=True,
      )
  )


if __name__ == "__main__":
  main()
