# PR Draft: Add Default-Off CenterPoint Proposal Refinement Extension

## Scope Audit

This draft is for the first conservative Apollo community PR only. If the local
development branch also contains proposal export tooling, student-model runtime
code, offline quantum/distillation tooling, downloaded models, records, or
skill files, split those changes out before opening this PR.

Recommended first-PR include list:

- `modules/perception/lidar_detection/detector/center_point_detection/proposal_refine.h`
- `modules/perception/lidar_detection/detector/center_point_detection/proposal_refine.cc`
- `modules/perception/lidar_detection/detector/center_point_detection/proposal_refine_test.cc`
- CenterPoint integration needed to call the default-off refine slot
- `ProposalRefineParam` config fields
- BUILD updates for the refine library/test
- this documentation

Recommended first-PR exclude list:

- proposal export files and reader
- `student_model` runtime integration
- offline quantum/distillation tools
- generated experiment reports
- downloaded model weights and records
- local `CODEX_SKILL.md` / `AGENTS.md`
- unrelated TensorRT or test-data setup fixes

## Title

perception: add default-off CenterPoint proposal refinement extension

## Summary

This PR adds an optional proposal refinement extension point to the LiDAR
CenterPoint detector. The extension is disabled by default and runs only when
`enable_proposal_refine=true`.

The initial implementation includes:

- `ProposalRefineModule`
- default-off config fields
- `mock_zero` no-op mode
- `mock_delta` deterministic test mode
- focused unit tests
- documentation and rollback instructions

This PR does not change Cyber RT channels, public perception message semantics,
tracking, fusion, prediction, or planning. It does not add PennyLane, Qiskit, or
any quantum runtime dependency.

## Motivation

CenterPoint currently goes directly from decoded proposals into the existing
post-processing path. A small, default-disabled extension point makes it
possible to experiment with proposal-level refinement while preserving the
original production behavior by default.

## Scope

Included:

- CenterPoint-internal optional proposal refinement slot
- default-off config
- proposal feature/delta data structures
- no-op and deterministic mock modes
- tests for disabled behavior, mock modes, score threshold, top-K selection,
  angle normalization, score clamping, and positive box dimensions

Excluded:

- quantum teacher code
- student model runtime
- proposal export tooling
- model weights
- datasets or records
- performance improvement claims

## Test Evidence

Checks run locally inside Apollo Docker:

```bash
bazel test --config=unit_test --cache_test_results=no --test_output=errors \
  //modules/perception/lidar_detection:proposal_refine_test

bazel build --config=opt --config=gpu --config=nvidia \
  --define ENABLE_PROFILER=true \
  --copt=-mavx2 --host_copt=-mavx2 \
  //modules/perception/lidar_detection:apollo_perception_lidar_detection \
  //modules/perception/lidar_detection:liblidar_detection_component.so

bazel build \
  //modules/perception/lidar_detection:center_point_proposal_refine_cpplint \
  //modules/perception/lidar_detection:proposal_refine_test_cpplint

git diff --check
```

Full `bash apollo.sh build`, `bash apollo.sh test`, `bash apollo.sh lint`, and
`bash apollo.sh check` were not run locally because they are broad repository
checks and substantially more expensive than the focused perception targets.
Expected CI coverage should include full build, test, lint, and check jobs.

## Default Behavior

The default config is:

```text
enable_proposal_refine=false
```

When disabled, the extension does not call `ProposalRefineModule::Refine` and
CenterPoint continues through the original object filling/NMS path.

## Risks

- The extension touches CenterPoint detector internals.
- Enabled mock modes can perturb proposals, so they must remain opt-in.
- This PR does not claim perception quality improvement.

Mitigations:

- disabled-by-default config
- focused unit tests
- no channel/message changes
- rollback by disabling `enable_proposal_refine`

## Rollback

Set `enable_proposal_refine=false` or remove the optional config block. No Cyber
RT channel or downstream module rollback is required.

## Follow-Up Work

Possible follow-up PRs:

- optional proposal export tool
- offline research tooling
- distilled student model integration, if separately reviewed and validated
- full record replay and labeled-dataset regression report
