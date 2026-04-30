# Phase 9 Proposal-Level Comparison

This report is a proposal-level regression report. It does not run full NMS/object filling and does not include GT labels, so it must not be used to claim detection performance improvement.

## Reproducibility

- Git branch: `master`
- Git commit: `d53aa3da47a06a08e6d0cd175d5623a34fa0d6aa`
- Apollo describe: `v11.0.0-10-gd53aa3da47-dirty`
- Dataset: `synthetic_proposals_seed_2026_frames_4_proposals_32`
- Command: `tools/quantum_centerpoint_refine/scripts/compare_refine_modes.py --synthetic --num-synthetic-frames 4 --proposals-per-frame 32 --seed 2026 --student-metadata /tmp/phase7_student_distill/student_metadata.json --student-model /tmp/phase7_student_distill/student_model.onnx --teacher-checkpoint /tmp/phase6_q_refine_train/q_refine_checkpoint.npz --output-dir tools/quantum_centerpoint_refine/results`
- Student model: `/tmp/phase7_student_distill/student_model.onnx`
- Student metadata: `/tmp/phase7_student_distill/student_metadata.json`
- Teacher checkpoint: `/tmp/phase6_q_refine_train/q_refine_checkpoint.npz`

## Metrics

| Mode | Proposals | Objects After NMS | Invalid Boxes | Latency Mean (ms) | Center Shift Mean | Size Shift Mean | Yaw Shift Mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| mock_delta | 128 | N/A | 0 | 0.054185 | 0.100000 | 0.000000 | 0.050000 |
| mock_zero | 128 | N/A | 0 | 0.014027 | 0.000000 | 0.000000 | 0.000000 |
| offline_quantum_teacher | 128 | N/A | 0 | 9.012879 | 0.005064 | 0.002612 | 0.004394 |
| original_centerpoint | 128 | N/A | 0 | 0.304453 | 0.000000 | 0.000000 | 0.000000 |
| student_model | 128 | N/A | 0 | 0.141774 | 0.265264 | 0.337904 | 0.143066 |

## Gate Checks

- mock_zero matches baseline: `True`
- student_model metrics reported: `True`
- latency overhead reported: `True`
- invalid boxes checked: `True`

GT metrics are unavailable because no labeled GT set was provided.
