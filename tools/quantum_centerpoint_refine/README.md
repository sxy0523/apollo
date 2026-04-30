# Hybrid Quantum CenterPoint Offline Refine Tool

This directory is an offline research tool for training a proposal refinement
teacher from CenterPoint proposal exports. It is intentionally separate from
Apollo runtime targets.

## Scope

- Reads proposal export files written by
  `modules/perception/lidar_detection/detector/center_point_detection/proposal_export.*`.
- Trains a small NumPy quantum-inspired teacher prototype.
- Keeps all quantum/research code outside Apollo C++ perception runtime.
- Does not modify Cyber RT channels, perception messages, tracking, fusion,
  prediction, planning, or CenterPoint default behavior.

## Proposal Feature Order

The Phase 5 exporter writes one binary file per frame. Each record is:

```text
[x, y, z, length, width, height, yaw, score, class_id]
```

The teacher predicts residuals in this order:

```text
[delta_x, delta_y, delta_z,
 delta_length, delta_width, delta_height,
 delta_yaw, delta_score]
```

BEV features are not exported yet. To train on BEV/grid features later, the
Apollo CenterPoint inference/decode path would need to expose or cache the
relevant BEV tensor and attach a stable grid index or feature vector to each
decoded proposal before NMS.

## Model

`QuantumRefineHead` is a pure NumPy prototype:

1. Linear projection from proposal feature dimension to `n_qubits`.
2. Angle encoding with `Ry` rotations.
3. `n_layers` rounds of trainable `Ry` rotations and ring CNOT entanglement.
4. Pauli-Z expectation measurement for each qubit.
5. Linear output head to 8 residual values.

The default config uses 4 qubits and 2 layers. No PennyLane or Qiskit dependency
is required.

## Quick Start

From the Apollo repository root:

```bash
python3 -m unittest discover -s tools/quantum_centerpoint_refine/tests

python3 tools/quantum_centerpoint_refine/scripts/train_q_refine.py \
  --synthetic \
  --num-synthetic 64 \
  --epochs 2 \
  --output-dir /tmp/q_refine_demo

python3 tools/quantum_centerpoint_refine/scripts/eval_q_refine.py \
  --checkpoint /tmp/q_refine_demo/q_refine_checkpoint.npz \
  --synthetic \
  --num-synthetic 32

python3 tools/quantum_centerpoint_refine/scripts/distill_student.py \
  --teacher-checkpoint /tmp/q_refine_demo/q_refine_checkpoint.npz \
  --synthetic \
  --num-synthetic 128 \
  --epochs 100 \
  --output-dir /tmp/q_refine_student
```

To train from exported proposals:

```bash
python3 tools/quantum_centerpoint_refine/scripts/train_q_refine.py \
  --export-path /apollo/data/perception/proposal_export \
  --output-dir /tmp/q_refine_from_export
```

If supervised residual targets are available, pass `--target-path` with a `.npy`
or CSV file containing one row per proposal and 8 columns in the residual order
shown above. Without supervised targets, the dataset uses deterministic
synthetic residual targets for pipeline testing.

The distillation script writes:

- `student_checkpoint.npz`
- `student_model.onnx`
- `student_metadata.json`
- `distill_log.json`

The ONNX input is normalized proposal features. The normalization mean/std and
expected Apollo refine config are written into `student_metadata.json`.

## Runtime Boundary

This tool must remain offline-only. Do not add it to Apollo runtime Bazel
targets and do not make Apollo perception depend on quantum Python packages.
