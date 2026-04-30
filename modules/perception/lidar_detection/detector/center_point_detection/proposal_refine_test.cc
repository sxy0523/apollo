/******************************************************************************
 * Copyright 2026 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "modules/perception/lidar_detection/detector/center_point_detection/proposal_refine.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "cyber/common/file.h"
#include "modules/common/math/math_utils.h"

namespace apollo {
namespace perception {
namespace lidar {
namespace {

constexpr double kPi = 3.14159265358979323846;

Proposal MakeProposal(float x, float score, int class_id = 0) {
  Proposal proposal;
  proposal.x = x;
  proposal.y = x + 1.0f;
  proposal.z = x + 2.0f;
  proposal.length = x + 3.0f;
  proposal.width = x + 4.0f;
  proposal.height = x + 5.0f;
  proposal.yaw = x + 6.0f;
  proposal.score = score;
  proposal.class_id = class_id;
  return proposal;
}

std::string TestDir(const std::string& name) {
  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  std::string base = test_tmpdir == nullptr ? "/tmp" : test_tmpdir;
  const int64_t suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::string dir = base + "/" + name + "_" + std::to_string(suffix);
  EXPECT_TRUE(apollo::cyber::common::EnsureDirectory(dir));
  return dir;
}

void WriteTextFile(const std::string& path, const std::string& content) {
  std::ofstream out(path, std::ios::out);
  ASSERT_TRUE(out.is_open());
  out << content;
}

void AppendVarint(uint64_t value, std::string* out) {
  while (value > 0x7f) {
    out->push_back(static_cast<char>((value & 0x7f) | 0x80));
    value >>= 7;
  }
  out->push_back(static_cast<char>(value));
}

void AppendKey(int field_number, int wire_type, std::string* out) {
  AppendVarint((static_cast<uint64_t>(field_number) << 3) | wire_type, out);
}

void AppendIntField(int field_number, uint64_t value, std::string* out) {
  AppendKey(field_number, 0, out);
  AppendVarint(value, out);
}

void AppendStringField(int field_number, const std::string& value,
                       std::string* out) {
  AppendKey(field_number, 2, out);
  AppendVarint(value.size(), out);
  out->append(value);
}

void AppendMessageField(int field_number, const std::string& value,
                        std::string* out) {
  AppendStringField(field_number, value, out);
}

std::string TensorProto(const std::string& name,
                        const std::vector<int64_t>& dims,
                        const std::vector<float>& values) {
  std::string tensor;
  for (const int64_t dim : dims) {
    AppendIntField(1, static_cast<uint64_t>(dim), &tensor);
  }
  AppendIntField(2, 1, &tensor);
  AppendStringField(8, name, &tensor);
  std::string raw(reinterpret_cast<const char*>(values.data()),
                  values.size() * sizeof(float));
  AppendStringField(9, raw, &tensor);
  return tensor;
}

void WriteSingleLayerStudentOnnx(const std::string& path,
                                 const std::vector<float>& weights,
                                 const std::vector<float>& bias) {
  std::string graph;
  AppendMessageField(5, TensorProto("weight_0", {9, 8}, weights), &graph);
  AppendMessageField(5, TensorProto("bias_0", {8}, bias), &graph);

  std::string model;
  AppendIntField(1, 7, &model);
  AppendStringField(2, "apollo_proposal_refine_test", &model);
  AppendMessageField(7, graph, &model);

  std::ofstream out(path, std::ios::binary | std::ios::out);
  ASSERT_TRUE(out.is_open());
  out.write(model.data(), model.size());
}

std::string JsonFloatArray(const std::vector<float>& values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string MetadataJson(
    const std::string& first_feature = "x",
    const std::vector<float>& mean = std::vector<float>(9, 0.0f),
    const std::vector<float>& std_values = std::vector<float>(9, 1.0f)) {
  std::ostringstream oss;
  oss << "{\n"
      << "\"input_dim\": 9,\n"
      << "\"output_dim\": 8,\n"
      << "\"feature_order\": [\"" << first_feature
      << "\",\"y\",\"z\",\"length\",\"width\",\"height\",\"yaw\","
      << "\"score\",\"class_id\"],\n"
      << "\"delta_order\": [\"delta_x\",\"delta_y\",\"delta_z\","
      << "\"delta_length\",\"delta_width\",\"delta_height\","
      << "\"delta_yaw\",\"delta_score\"],\n"
      << "\"normalization\": {\"mean\": " << JsonFloatArray(mean)
      << ", \"std\": " << JsonFloatArray(std_values) << "}\n"
      << "}\n";
  return oss.str();
}

void ExpectProposalEq(const Proposal& lhs, const Proposal& rhs) {
  EXPECT_FLOAT_EQ(lhs.x, rhs.x);
  EXPECT_FLOAT_EQ(lhs.y, rhs.y);
  EXPECT_FLOAT_EQ(lhs.z, rhs.z);
  EXPECT_FLOAT_EQ(lhs.length, rhs.length);
  EXPECT_FLOAT_EQ(lhs.width, rhs.width);
  EXPECT_FLOAT_EQ(lhs.height, rhs.height);
  EXPECT_FLOAT_EQ(lhs.yaw, rhs.yaw);
  EXPECT_FLOAT_EQ(lhs.score, rhs.score);
  EXPECT_EQ(lhs.class_id, rhs.class_id);
}

}  // namespace

TEST(ProposalRefineMathTest, NormalizeAngleBoundaries) {
  EXPECT_DOUBLE_EQ(-kPi, apollo::common::math::NormalizeAngle(kPi));
  EXPECT_DOUBLE_EQ(-kPi, apollo::common::math::NormalizeAngle(-kPi));
  EXPECT_NEAR(0.25, apollo::common::math::NormalizeAngle(4.0 * kPi + 0.25),
              1.0e-12);
}

TEST(ProposalRefineMathTest, ClampScore) {
  EXPECT_FLOAT_EQ(0.0f, apollo::common::math::Clamp(-0.2f, 0.0f, 1.0f));
  EXPECT_FLOAT_EQ(0.4f, apollo::common::math::Clamp(0.4f, 0.0f, 1.0f));
  EXPECT_FLOAT_EQ(1.0f, apollo::common::math::Clamp(1.2f, 0.0f, 1.0f));
}

TEST(ProposalRefineMathTest, PositiveDimensions) {
  Proposal proposal = MakeProposal(1.0f, 0.5f);
  proposal.length = 0.2f;
  proposal.width = 0.3f;
  proposal.height = 0.4f;
  ProposalDelta delta;
  delta.delta_length = -10.0f;
  delta.delta_width = -10.0f;
  delta.delta_height = -10.0f;

  ApplyProposalDelta(delta, 1.0f, &proposal);

  EXPECT_GT(proposal.length, 0.0f);
  EXPECT_GT(proposal.width, 0.0f);
  EXPECT_GT(proposal.height, 0.0f);
}

TEST(ProposalRefineModuleTest, DisabledPathDoesNotCallRefine) {
  centerpoint::ProposalRefineParam config;

  ProposalRefineModule module;
  ASSERT_TRUE(module.Init(config));
  ASSERT_FALSE(module.enabled());

  std::vector<Proposal> proposals = {MakeProposal(1.0f, 0.9f)};
  const std::vector<Proposal> original = proposals;
  ProposalRefineStats stats;
  ASSERT_TRUE(MaybeRefineProposals(&module, &proposals, &stats));

  EXPECT_EQ(0, module.refine_call_count());
  EXPECT_EQ(0U, stats.proposal_count);
  EXPECT_EQ(0U, stats.refined_proposal_count);
  ASSERT_EQ(original.size(), proposals.size());
  ExpectProposalEq(original[0], proposals[0]);
}

TEST(ProposalRefineModuleTest, MockZeroLeavesProposalsUnchanged) {
  centerpoint::ProposalRefineParam config;
  config.set_enable_proposal_refine(true);
  config.set_proposal_refine_mode("mock_zero");

  ProposalRefineModule module;
  ASSERT_TRUE(module.Init(config));

  std::vector<Proposal> proposals = {
      MakeProposal(1.0f, 0.7f, 0),
      MakeProposal(2.0f, 0.8f, 1),
  };
  const std::vector<Proposal> original = proposals;
  ProposalRefineStats stats;
  ASSERT_TRUE(MaybeRefineProposals(&module, &proposals, &stats));

  EXPECT_EQ(1, module.refine_call_count());
  EXPECT_EQ(2U, stats.proposal_count);
  EXPECT_EQ(2U, stats.refined_proposal_count);
  EXPECT_FLOAT_EQ(0.0f, stats.delta_min);
  EXPECT_FLOAT_EQ(0.0f, stats.delta_max);
  ASSERT_EQ(original.size(), proposals.size());
  for (size_t i = 0; i < proposals.size(); ++i) {
    ExpectProposalEq(original[i], proposals[i]);
  }
}

TEST(ProposalRefineModuleTest, MockDeltaChangesProposalsDeterministically) {
  centerpoint::ProposalRefineParam config;
  config.set_enable_proposal_refine(true);
  config.set_proposal_refine_mode("mock_delta");

  ProposalRefineModule module;
  ASSERT_TRUE(module.Init(config));

  std::vector<Proposal> proposals = {MakeProposal(1.0f, 0.7f, 0)};
  proposals[0].yaw = 0.2f;
  const Proposal original = proposals[0];
  ProposalRefineStats stats;
  ASSERT_TRUE(MaybeRefineProposals(&module, &proposals, &stats));

  EXPECT_EQ(1, module.refine_call_count());
  EXPECT_EQ(1U, stats.proposal_count);
  EXPECT_EQ(1U, stats.refined_proposal_count);
  EXPECT_FLOAT_EQ(0.0f, stats.delta_min);
  EXPECT_FLOAT_EQ(0.1f, stats.delta_max);
  EXPECT_FLOAT_EQ(original.x + 0.1f, proposals[0].x);
  EXPECT_FLOAT_EQ(original.y, proposals[0].y);
  EXPECT_FLOAT_EQ(original.z, proposals[0].z);
  EXPECT_FLOAT_EQ(original.length, proposals[0].length);
  EXPECT_FLOAT_EQ(original.width, proposals[0].width);
  EXPECT_FLOAT_EQ(original.height, proposals[0].height);
  EXPECT_NEAR(original.yaw + 0.05f, proposals[0].yaw, 1.0e-6f);
  EXPECT_FLOAT_EQ(original.score + 0.01f, proposals[0].score);
}

TEST(ProposalRefineModuleTest, MockDeltaWrapsYawAndClampsScore) {
  centerpoint::ProposalRefineParam config;
  config.set_enable_proposal_refine(true);
  config.set_proposal_refine_mode("mock_delta");

  ProposalRefineModule module;
  ASSERT_TRUE(module.Init(config));

  std::vector<Proposal> proposals = {MakeProposal(1.0f, 0.995f, 0)};
  proposals[0].yaw = static_cast<float>(kPi - 0.02);
  ProposalRefineStats stats;
  ASSERT_TRUE(MaybeRefineProposals(&module, &proposals, &stats));

  const double expected_yaw =
      apollo::common::math::NormalizeAngle(kPi - 0.02 + 0.05);
  EXPECT_NEAR(expected_yaw, proposals[0].yaw, 1.0e-6);
  EXPECT_FLOAT_EQ(1.0f, proposals[0].score);
}

TEST(ProposalRefineModuleTest, MockDeltaRespectsThresholdAndTopK) {
  centerpoint::ProposalRefineParam config;
  config.set_enable_proposal_refine(true);
  config.set_proposal_refine_mode("mock_delta");
  config.set_proposal_refine_score_threshold(0.5f);
  config.set_proposal_refine_top_k(1);

  ProposalRefineModule module;
  ASSERT_TRUE(module.Init(config));

  std::vector<Proposal> proposals = {
      MakeProposal(1.0f, 0.9f),
      MakeProposal(2.0f, 0.8f),
      MakeProposal(3.0f, 0.4f),
  };
  const std::vector<Proposal> original = proposals;
  ProposalRefineStats stats;
  ASSERT_TRUE(MaybeRefineProposals(&module, &proposals, &stats));

  EXPECT_EQ(3U, stats.proposal_count);
  EXPECT_EQ(1U, stats.refined_proposal_count);
  EXPECT_FLOAT_EQ(original[0].x + 0.1f, proposals[0].x);
  ExpectProposalEq(original[1], proposals[1]);
  ExpectProposalEq(original[2], proposals[2]);
}

TEST(ProposalRefineModuleTest, EmptyProposalsDoNotCrash) {
  centerpoint::ProposalRefineParam config;
  config.set_enable_proposal_refine(true);
  config.set_proposal_refine_mode("mock_zero");

  ProposalRefineModule module;
  ASSERT_TRUE(module.Init(config));

  std::vector<Proposal> proposals;
  ProposalRefineStats stats;
  ASSERT_TRUE(MaybeRefineProposals(&module, &proposals, &stats));
  EXPECT_EQ(1, module.refine_call_count());
  EXPECT_EQ(0U, stats.proposal_count);
  EXPECT_EQ(0U, stats.refined_proposal_count);
  EXPECT_TRUE(proposals.empty());
}

TEST(ProposalRefineModuleTest, ThresholdFilteringControlsRefineCount) {
  centerpoint::ProposalRefineParam config;
  config.set_enable_proposal_refine(true);
  config.set_proposal_refine_mode("mock_zero");
  config.set_proposal_refine_score_threshold(0.5f);

  ProposalRefineModule module;
  ASSERT_TRUE(module.Init(config));

  std::vector<Proposal> proposals = {
      MakeProposal(1.0f, 0.49f),
      MakeProposal(2.0f, 0.5f),
      MakeProposal(3.0f, 0.9f),
  };
  ProposalRefineStats stats;
  ASSERT_TRUE(MaybeRefineProposals(&module, &proposals, &stats));
  EXPECT_EQ(3U, stats.proposal_count);
  EXPECT_EQ(2U, stats.refined_proposal_count);
}

TEST(ProposalRefineModuleTest, TopKLimitsRefineCount) {
  centerpoint::ProposalRefineParam config;
  config.set_enable_proposal_refine(true);
  config.set_proposal_refine_mode("mock_zero");
  config.set_proposal_refine_top_k(2);

  ProposalRefineModule module;
  ASSERT_TRUE(module.Init(config));

  std::vector<Proposal> proposals = {
      MakeProposal(1.0f, 0.1f),
      MakeProposal(2.0f, 0.9f),
      MakeProposal(3.0f, 0.8f),
      MakeProposal(4.0f, 0.7f),
  };
  ProposalRefineStats stats;
  ASSERT_TRUE(MaybeRefineProposals(&module, &proposals, &stats));
  EXPECT_EQ(4U, stats.proposal_count);
  EXPECT_EQ(2U, stats.refined_proposal_count);
}

TEST(ProposalRefineModuleTest, UnsupportedEnabledModeFailsInit) {
  centerpoint::ProposalRefineParam config;
  config.set_enable_proposal_refine(true);
  config.set_proposal_refine_mode("unknown_mode");

  ProposalRefineModule module;
  EXPECT_FALSE(module.Init(config));
}

TEST(ProposalRefineModuleTest, StudentModelMissingFileDisablesWhenNotStrict) {
  const std::string dir = TestDir("student_model_missing_fallback");
  const std::string metadata_path = dir + "/student_metadata.json";
  WriteTextFile(metadata_path, MetadataJson());

  centerpoint::ProposalRefineParam config;
  config.set_enable_proposal_refine(true);
  config.set_proposal_refine_mode("student_model");
  config.set_proposal_refine_model_path(dir + "/missing.onnx");
  config.set_proposal_refine_metadata_path(metadata_path);
  config.set_strict_refine(false);

  ProposalRefineModule module;
  ASSERT_TRUE(module.Init(config));
  EXPECT_FALSE(module.enabled());

  std::vector<Proposal> proposals = {MakeProposal(1.0f, 0.9f)};
  const std::vector<Proposal> original = proposals;
  ProposalRefineStats stats;
  ASSERT_TRUE(MaybeRefineProposals(&module, &proposals, &stats));
  EXPECT_EQ(0, module.refine_call_count());
  ExpectProposalEq(original[0], proposals[0]);
}

TEST(ProposalRefineModuleTest, StudentModelMissingFileFailsWhenStrict) {
  const std::string dir = TestDir("student_model_missing_strict");
  const std::string metadata_path = dir + "/student_metadata.json";
  WriteTextFile(metadata_path, MetadataJson());

  centerpoint::ProposalRefineParam config;
  config.set_enable_proposal_refine(true);
  config.set_proposal_refine_mode("student_model");
  config.set_proposal_refine_model_path(dir + "/missing.onnx");
  config.set_proposal_refine_metadata_path(metadata_path);
  config.set_strict_refine(true);

  ProposalRefineModule module;
  EXPECT_FALSE(module.Init(config));
}

TEST(ProposalRefineModuleTest, StudentModelInvalidFeatureOrderFailsInit) {
  const std::string dir = TestDir("student_model_bad_feature_order");
  const std::string metadata_path = dir + "/student_metadata.json";
  const std::string model_path = dir + "/student_model.onnx";
  WriteTextFile(metadata_path, MetadataJson("bad_x"));
  WriteSingleLayerStudentOnnx(model_path, std::vector<float>(9 * 8, 0.0f),
                              std::vector<float>(8, 0.0f));

  centerpoint::ProposalRefineParam config;
  config.set_enable_proposal_refine(true);
  config.set_proposal_refine_mode("student_model");
  config.set_proposal_refine_model_path(model_path);
  config.set_proposal_refine_metadata_path(metadata_path);
  config.set_strict_refine(false);

  ProposalRefineModule module;
  EXPECT_FALSE(module.Init(config));
}

TEST(ProposalRefineModuleTest, StudentModelAppliesNormalizationAndDelta) {
  const std::string dir = TestDir("student_model_runs");
  const std::string metadata_path = dir + "/student_metadata.json";
  const std::string model_path = dir + "/student_model.onnx";
  std::vector<float> mean(9, 0.0f);
  std::vector<float> std(9, 1.0f);
  mean[0] = 10.0f;
  std[0] = 2.0f;
  WriteTextFile(metadata_path, MetadataJson("x", mean, std));
  std::vector<float> weights(9 * 8, 0.0f);
  weights[0 * 8 + 0] = 0.25f;
  WriteSingleLayerStudentOnnx(model_path, weights, std::vector<float>(8, 0.0f));

  centerpoint::ProposalRefineParam config;
  config.set_enable_proposal_refine(true);
  config.set_proposal_refine_mode("student_model");
  config.set_proposal_refine_model_path(model_path);
  config.set_proposal_refine_metadata_path(metadata_path);
  config.set_proposal_refine_top_k(1);
  config.set_proposal_refine_score_threshold(0.5f);

  ProposalRefineModule module;
  ASSERT_TRUE(module.Init(config));
  ASSERT_TRUE(module.enabled());

  std::vector<Proposal> proposals = {
      MakeProposal(14.0f, 0.9f),
      MakeProposal(18.0f, 0.8f),
      MakeProposal(30.0f, 0.4f),
  };
  const std::vector<Proposal> original = proposals;
  ProposalRefineStats stats;
  ASSERT_TRUE(MaybeRefineProposals(&module, &proposals, &stats));

  EXPECT_EQ(1, module.refine_call_count());
  EXPECT_EQ(3U, stats.proposal_count);
  EXPECT_EQ(1U, stats.refined_proposal_count);
  EXPECT_FLOAT_EQ(0.0f, stats.delta_min);
  EXPECT_FLOAT_EQ(0.5f, stats.delta_max);
  EXPECT_GE(stats.feature_build_latency_ms, 0.0);
  EXPECT_GE(stats.inference_latency_ms, 0.0);
  EXPECT_GE(stats.delta_apply_latency_ms, 0.0);
  EXPECT_FLOAT_EQ(original[0].x + 0.5f, proposals[0].x);
  ExpectProposalEq(original[1], proposals[1]);
  ExpectProposalEq(original[2], proposals[2]);
}

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
