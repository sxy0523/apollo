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

"""Creates a proposal-level comparison report for refine modes.

The report is intentionally proposal-level. It can check default-path parity,
student_model deltas, latency, and invalid boxes from exported proposals without
claiming GT/NMS quality improvements.
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from datasets.proposal_dataset import (  # pylint: disable=wrong-import-position
    FEATURE_ORDER,
    load_proposal_exports,
    make_synthetic_features,
)
from models.quantum_refine_head import (  # pylint: disable=wrong-import-position
    QuantumRefineHead,
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


def _run_git(args):
  try:
    return subprocess.check_output(
        ["git"] + args, cwd=REPO_ROOT, stderr=subprocess.DEVNULL
    ).decode("utf-8").strip()
  except Exception:
    return ""


def _read_varint(data, offset):
  shift = 0
  value = 0
  while offset < len(data):
    byte = data[offset]
    if not isinstance(byte, int):
      byte = ord(byte)
    offset += 1
    value |= (byte & 0x7F) << shift
    if not byte & 0x80:
      return value, offset
    shift += 7
  raise ValueError("truncated varint")


def _iter_fields(data):
  offset = 0
  while offset < len(data):
    key, offset = _read_varint(data, offset)
    field_number = key >> 3
    wire_type = key & 0x07
    if wire_type == 0:
      value, offset = _read_varint(data, offset)
      yield field_number, wire_type, value
    elif wire_type == 2:
      length, offset = _read_varint(data, offset)
      value = data[offset:offset + length]
      offset += length
      yield field_number, wire_type, value
    else:
      raise ValueError("unsupported protobuf wire type: %d" % wire_type)


def _parse_tensor(data):
  dims = []
  name = ""
  raw_data = b""
  for field_number, _, value in _iter_fields(data):
    if field_number == 1:
      dims.append(int(value))
    elif field_number == 8:
      name = value.decode("utf-8")
    elif field_number == 9:
      raw_data = value
  if not dims or not name or not raw_data:
    raise ValueError("unsupported ONNX tensor initializer")
  return name, np.frombuffer(raw_data, dtype="<f4").reshape(dims)


def _read_onnx_initializers(path):
  with open(path, "rb") as model_file:
    model = model_file.read()
  graph = None
  for field_number, _, value in _iter_fields(model):
    if field_number == 7:
      graph = value
      break
  if graph is None:
    raise ValueError("ONNX graph is missing")

  tensors = {}
  for field_number, _, value in _iter_fields(graph):
    if field_number == 5:
      name, tensor = _parse_tensor(value)
      tensors[name] = tensor.astype(np.float32)
  return tensors


class StudentRuntime:
  """Small NumPy runtime matching the Phase 8 C++ student_model path."""

  def __init__(self, metadata_path, onnx_path):
    with open(metadata_path) as metadata_file:
      self.metadata = json.load(metadata_file)
    self.mean = np.asarray(
        self.metadata["normalization"]["mean"], dtype=np.float32)
    self.std = np.asarray(
        self.metadata["normalization"]["std"], dtype=np.float32)
    self.tensors = _read_onnx_initializers(onnx_path)
    self.layers = []
    index = 0
    while "weight_%d" % index in self.tensors:
      self.layers.append((
          self.tensors["weight_%d" % index],
          self.tensors["bias_%d" % index],
      ))
      index += 1
    if not self.layers:
      raise ValueError("student ONNX has no MLP layers")

  def __call__(self, features):
    values = ((features - self.mean) / self.std).astype(np.float32)
    for index, (weight, bias) in enumerate(self.layers):
      values = values @ weight + bias
      if index + 1 < len(self.layers):
        values = np.maximum(values, 0.0)
    return values.astype(np.float32)


def _normalize_angle(values):
  return (values + np.pi) % (2.0 * np.pi) - np.pi


def _apply_deltas(features, deltas, delta_scale):
  refined = features.copy()
  scaled = deltas * delta_scale
  refined[:, 0:7] += scaled[:, 0:7]
  refined[:, 3:6] = np.maximum(refined[:, 3:6], 1.0e-3)
  refined[:, 6] = _normalize_angle(refined[:, 6])
  refined[:, 7] = np.clip(refined[:, 7] + scaled[:, 7], 0.0, 1.0)
  return refined.astype(np.float32)


def _select_indices(features, score_threshold, top_k):
  selected = np.where(features[:, 7] >= score_threshold)[0]
  order = np.argsort(-features[selected, 7], kind="stable")
  selected = selected[order]
  if top_k >= 0:
    selected = selected[:top_k]
  return selected


def _apply_mode(features, mode, args, student_runtime, teacher):
  refined = features.copy()
  selected = _select_indices(features, args.score_threshold, args.top_k)
  if selected.size == 0:
    return refined

  if mode == "original_centerpoint" or mode == "mock_zero":
    return refined
  if mode == "mock_delta":
    deltas = np.zeros((selected.size, len(DELTA_ORDER)), dtype=np.float32)
    deltas[:, 0] = 0.1
    deltas[:, 6] = 0.05
    deltas[:, 7] = 0.01
  elif mode == "student_model":
    deltas = student_runtime(features[selected])
  elif mode == "offline_quantum_teacher":
    deltas = teacher(features[selected])
  else:
    raise ValueError("unknown mode: %s" % mode)
  refined[selected] = _apply_deltas(
      refined[selected], deltas.astype(np.float32), args.delta_scale)
  return refined


def _percentile(values, pct):
  if values.size == 0:
    return None
  return float(np.percentile(values, pct))


def _safe_float(value):
  if value is None:
    return None
  return float(value)


def _distribution(values):
  values = np.asarray(values, dtype=np.float32)
  if values.size == 0:
    return {"min": None, "mean": None, "p50": None, "p95": None, "max": None}
  return {
      "min": float(np.min(values)),
      "mean": float(np.mean(values)),
      "p50": _percentile(values, 50),
      "p95": _percentile(values, 95),
      "max": float(np.max(values)),
  }


def _class_distribution(features):
  result = {}
  if features.size == 0:
    return result
  classes, counts = np.unique(features[:, 8].astype(np.int32), return_counts=True)
  for class_id, count in zip(classes, counts):
    result[str(int(class_id))] = int(count)
  return result


def _invalid_box_count(features):
  if features.size == 0:
    return 0
  finite = np.all(np.isfinite(features), axis=1)
  positive_size = np.all(features[:, 3:6] > 0.0, axis=1)
  score_ok = (features[:, 7] >= 0.0) & (features[:, 7] <= 1.0)
  return int(np.sum(~(finite & positive_size & score_ok)))


def _frame_metrics(frame_index, mode, baseline, refined, latency_ms):
  center_shift = np.linalg.norm(refined[:, 0:3] - baseline[:, 0:3], axis=1)
  size_shift = np.linalg.norm(refined[:, 3:6] - baseline[:, 3:6], axis=1)
  yaw_shift = np.abs(_normalize_angle(refined[:, 6] - baseline[:, 6]))
  score_shift = np.abs(refined[:, 7] - baseline[:, 7])
  return {
      "frame_index": frame_index,
      "mode": mode,
      "proposal_count_before_nms": int(baseline.shape[0]),
      "object_count_after_nms": "",
      "class_distribution": json.dumps(_class_distribution(refined),
                                       sort_keys=True),
      "score_min": _safe_float(_distribution(refined[:, 7])["min"]),
      "score_mean": _safe_float(_distribution(refined[:, 7])["mean"]),
      "score_max": _safe_float(_distribution(refined[:, 7])["max"]),
      "center_shift_mean": _safe_float(_distribution(center_shift)["mean"]),
      "center_shift_p95": _safe_float(_distribution(center_shift)["p95"]),
      "size_shift_mean": _safe_float(_distribution(size_shift)["mean"]),
      "size_shift_p95": _safe_float(_distribution(size_shift)["p95"]),
      "yaw_shift_mean": _safe_float(_distribution(yaw_shift)["mean"]),
      "yaw_shift_p95": _safe_float(_distribution(yaw_shift)["p95"]),
      "score_shift_mean": _safe_float(_distribution(score_shift)["mean"]),
      "latency_ms": float(latency_ms),
      "invalid_box_count": _invalid_box_count(refined),
  }


def _aggregate(rows):
  by_mode = {}
  for row in rows:
    by_mode.setdefault(row["mode"], []).append(row)

  summary = {}
  for mode, mode_rows in by_mode.items():
    latency = np.asarray([row["latency_ms"] for row in mode_rows],
                         dtype=np.float32)
    center = np.asarray([row["center_shift_mean"] for row in mode_rows],
                        dtype=np.float32)
    size = np.asarray([row["size_shift_mean"] for row in mode_rows],
                      dtype=np.float32)
    yaw = np.asarray([row["yaw_shift_mean"] for row in mode_rows],
                     dtype=np.float32)
    proposals = sum(row["proposal_count_before_nms"] for row in mode_rows)
    invalid = sum(row["invalid_box_count"] for row in mode_rows)
    class_counts = {}
    for row in mode_rows:
      for class_id, count in json.loads(row["class_distribution"]).items():
        class_counts[class_id] = class_counts.get(class_id, 0) + int(count)
    summary[mode] = {
        "frame_count": len(mode_rows),
        "proposal_count_before_nms": proposals,
        "object_count_after_nms": None,
        "object_count_after_nms_note": "not available in offline proposal report",
        "class_distribution": class_counts,
        "latency_ms": _distribution(latency),
        "center_shift_mean_distribution": _distribution(center),
        "size_shift_mean_distribution": _distribution(size),
        "yaw_shift_mean_distribution": _distribution(yaw),
        "invalid_box_count": invalid,
        "gt_metrics": None,
        "gt_metrics_note": "ground truth labels were not provided",
    }
  return summary


def _load_frames(args):
  if args.synthetic or not args.proposal_export_path:
    features = make_synthetic_features(
        args.num_synthetic_frames * args.proposals_per_frame, seed=args.seed)
    return [
        features[i * args.proposals_per_frame:(i + 1) * args.proposals_per_frame]
        for i in range(args.num_synthetic_frames)
    ], "synthetic_proposals_seed_%d_frames_%d_proposals_%d" % (
        args.seed, args.num_synthetic_frames, args.proposals_per_frame)

  features = load_proposal_exports(args.proposal_export_path)
  return [features], "proposal_export:%s" % args.proposal_export_path


def _write_csv(path, rows):
  fieldnames = [
      "frame_index",
      "mode",
      "proposal_count_before_nms",
      "object_count_after_nms",
      "class_distribution",
      "score_min",
      "score_mean",
      "score_max",
      "center_shift_mean",
      "center_shift_p95",
      "size_shift_mean",
      "size_shift_p95",
      "yaw_shift_mean",
      "yaw_shift_p95",
      "score_shift_mean",
      "latency_ms",
      "invalid_box_count",
  ]
  with open(path, "w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def _write_markdown(path, report):
  summary = report["summary"]
  lines = [
      "# Phase 9 Proposal-Level Comparison",
      "",
      "This report is a proposal-level regression report. It does not run "
      "full NMS/object filling and does not include GT labels, so it must not "
      "be used to claim detection performance improvement.",
      "",
      "## Reproducibility",
      "",
      "- Git branch: `%s`" % report["git"]["branch"],
      "- Git commit: `%s`" % report["git"]["commit"],
      "- Apollo describe: `%s`" % report["git"]["describe"],
      "- Dataset: `%s`" % report["dataset_identifier"],
      "- Command: `%s`" % report["command"],
      "- Student model: `%s`" % report["model_files"].get("student_model", ""),
      "- Student metadata: `%s`" % report["model_files"].get(
          "student_metadata", ""),
      "- Teacher checkpoint: `%s`" % report["model_files"].get(
          "teacher_checkpoint", ""),
      "",
      "## Metrics",
      "",
      "| Mode | Proposals | Objects After NMS | Invalid Boxes | "
      "Latency Mean (ms) | Center Shift Mean | Size Shift Mean | "
      "Yaw Shift Mean |",
      "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
  ]
  for mode in sorted(summary):
    item = summary[mode]
    lines.append(
        "| %s | %d | %s | %d | %.6f | %.6f | %.6f | %.6f |" % (
            mode,
            item["proposal_count_before_nms"],
            "N/A",
            item["invalid_box_count"],
            item["latency_ms"]["mean"] or 0.0,
            item["center_shift_mean_distribution"]["mean"] or 0.0,
            item["size_shift_mean_distribution"]["mean"] or 0.0,
            item["yaw_shift_mean_distribution"]["mean"] or 0.0,
        )
    )
  lines += [
      "",
      "## Gate Checks",
      "",
      "- mock_zero matches baseline: `%s`" % report["gate_checks"][
          "mock_zero_matches_baseline"],
      "- student_model metrics reported: `%s`" % report["gate_checks"][
          "student_model_metrics_reported"],
      "- latency overhead reported: `%s`" % report["gate_checks"][
          "latency_overhead_reported"],
      "- invalid boxes checked: `%s`" % report["gate_checks"][
          "invalid_boxes_checked"],
      "",
      "GT metrics are unavailable because no labeled GT set was provided.",
  ]
  with open(path, "w") as markdown_file:
    markdown_file.write("\n".join(lines) + "\n")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--proposal-export-path", default="")
  parser.add_argument("--synthetic", action="store_true")
  parser.add_argument("--num-synthetic-frames", type=int, default=4)
  parser.add_argument("--proposals-per-frame", type=int, default=32)
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--student-metadata", default="")
  parser.add_argument("--student-model", default="")
  parser.add_argument("--teacher-checkpoint", default="")
  parser.add_argument("--output-dir", default=os.path.join(ROOT, "results"))
  parser.add_argument("--top-k", type=int, default=-1)
  parser.add_argument("--score-threshold", type=float, default=0.0)
  parser.add_argument("--delta-scale", type=float, default=1.0)
  args = parser.parse_args()

  frames, dataset_identifier = _load_frames(args)
  student_runtime = None
  teacher = None
  modes = ["original_centerpoint", "mock_zero", "mock_delta"]
  if args.student_metadata and args.student_model:
    student_runtime = StudentRuntime(args.student_metadata, args.student_model)
    modes.append("student_model")
  if args.teacher_checkpoint:
    teacher = QuantumRefineHead.load_checkpoint(args.teacher_checkpoint)
    modes.append("offline_quantum_teacher")

  rows = []
  refined_by_mode = {mode: [] for mode in modes}
  for frame_index, baseline in enumerate(frames):
    baseline = baseline.astype(np.float32)
    for mode in modes:
      start = time.perf_counter()
      refined = _apply_mode(baseline, mode, args, student_runtime, teacher)
      latency_ms = (time.perf_counter() - start) * 1000.0
      refined_by_mode[mode].append(refined)
      rows.append(_frame_metrics(frame_index, mode, baseline, refined,
                                 latency_ms))

  summary = _aggregate(rows)
  mock_zero_matches = all(
      np.allclose(base, zero, atol=0.0)
      for base, zero in zip(refined_by_mode["original_centerpoint"],
                            refined_by_mode["mock_zero"]))
  gate_checks = {
      "mock_zero_matches_baseline": bool(mock_zero_matches),
      "student_model_metrics_reported": "student_model" in summary,
      "latency_overhead_reported": all(
          "latency_ms" in item and item["latency_ms"]["mean"] is not None
          for item in summary.values()),
      "invalid_boxes_checked": all(
          "invalid_box_count" in item for item in summary.values()),
  }

  report = {
      "git": {
          "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
          "commit": _run_git(["rev-parse", "HEAD"]),
          "describe": _run_git(["describe", "--tags", "--always", "--dirty"]),
      },
      "apollo_version_or_branch": _run_git(
          ["describe", "--tags", "--always", "--dirty"]),
      "dataset_identifier": dataset_identifier,
      "config_files": {
          "phase9_script": os.path.relpath(__file__, REPO_ROOT),
          "student_metadata": args.student_metadata,
      },
      "model_files": {
          "student_model": args.student_model,
          "student_metadata": args.student_metadata,
          "teacher_checkpoint": args.teacher_checkpoint,
      },
      "command": " ".join(sys.argv),
      "environment": {
          "platform": platform.platform(),
          "python": platform.python_version(),
          "numpy": np.__version__,
      },
      "modes": modes,
      "feature_order": list(FEATURE_ORDER),
      "delta_order": list(DELTA_ORDER),
      "summary": summary,
      "gate_checks": gate_checks,
      "notes": [
          "No performance improvement is claimed.",
          "Object-after-NMS and GT metrics require a full labeled replay run.",
      ],
  }

  os.makedirs(args.output_dir, exist_ok=True)
  summary_path = os.path.join(args.output_dir, "summary.json")
  csv_path = os.path.join(args.output_dir, "per_frame.csv")
  markdown_path = os.path.join(args.output_dir, "comparison.md")
  with open(summary_path, "w") as summary_file:
    json.dump(report, summary_file, indent=2, sort_keys=True)
  _write_csv(csv_path, rows)
  _write_markdown(markdown_path, report)
  print(json.dumps({
      "summary": summary_path,
      "per_frame": csv_path,
      "comparison": markdown_path,
      "gate_checks": gate_checks,
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
