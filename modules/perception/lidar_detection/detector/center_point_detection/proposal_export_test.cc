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

#include "modules/perception/lidar_detection/detector/center_point_detection/proposal_export.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "cyber/common/file.h"

namespace apollo {
namespace perception {
namespace lidar {
namespace {

template <typename T>
T ReadValue(std::ifstream* in) {
  T value;
  in->read(reinterpret_cast<char*>(&value), sizeof(T));
  return value;
}

struct ReadExportFileResult {
  ProposalExportHeader header;
  std::vector<ProposalExportRecord> records;
};

std::string TestDir(const std::string& name) {
  const char* test_tmpdir = std::getenv("TEST_TMPDIR");
  std::string base = test_tmpdir == nullptr ? "/tmp" : test_tmpdir;
  const int64_t suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  std::string dir = base + "/" + name + "_" + std::to_string(suffix);
  EXPECT_TRUE(apollo::cyber::common::EnsureDirectory(dir));
  return dir;
}

Proposal MakeProposal(float x, float score, int class_id) {
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

ReadExportFileResult ReadExportFile(const std::string& file_path) {
  std::ifstream in(file_path, std::ios::binary | std::ios::in);
  EXPECT_TRUE(in.is_open());
  ReadExportFileResult result;
  in.read(result.header.magic, sizeof(result.header.magic));
  result.header.version = ReadValue<uint32_t>(&in);
  result.header.record_size = ReadValue<uint32_t>(&in);
  result.header.sequence_id = ReadValue<uint64_t>(&in);
  result.header.timestamp = ReadValue<double>(&in);
  result.header.proposal_count = ReadValue<uint32_t>(&in);
  result.header.feature_dim = ReadValue<uint32_t>(&in);
  for (uint32_t i = 0; i < result.header.proposal_count; ++i) {
    ProposalExportRecord record;
    record.x = ReadValue<float>(&in);
    record.y = ReadValue<float>(&in);
    record.z = ReadValue<float>(&in);
    record.length = ReadValue<float>(&in);
    record.width = ReadValue<float>(&in);
    record.height = ReadValue<float>(&in);
    record.yaw = ReadValue<float>(&in);
    record.score = ReadValue<float>(&in);
    record.class_id = ReadValue<int32_t>(&in);
    result.records.push_back(record);
  }
  return result;
}

std::string FirstBinaryPath(const std::string& dir) {
  return dir + "/" + ProposalExportModule::BinaryFileName(0);
}

}  // namespace

TEST(ProposalExportModuleTest, DisabledProducesNoFile) {
  const std::string dir = TestDir("proposal_export_disabled");
  centerpoint::ProposalExportParam config;
  config.set_proposal_export_path(dir);

  ProposalExportModule module;
  ASSERT_TRUE(module.Init(config));
  EXPECT_TRUE(module.Export(1.0, {MakeProposal(1.0f, 0.5f, 7)}));
  EXPECT_FALSE(apollo::cyber::common::PathExists(FirstBinaryPath(dir)));
}

TEST(ProposalExportModuleTest, EmptyProposalExport) {
  const std::string dir = TestDir("proposal_export_empty");
  centerpoint::ProposalExportParam config;
  config.set_enable_proposal_export(true);
  config.set_proposal_export_path(dir);

  ProposalExportModule module;
  ASSERT_TRUE(module.Init(config));
  ASSERT_TRUE(module.Export(10.5, {}));

  const ReadExportFileResult result = ReadExportFile(FirstBinaryPath(dir));
  EXPECT_EQ(std::string("APCPROP1", 8), std::string(result.header.magic, 8));
  EXPECT_EQ(1U, result.header.version);
  EXPECT_EQ(sizeof(ProposalExportRecord), result.header.record_size);
  EXPECT_EQ(0U, result.header.sequence_id);
  EXPECT_DOUBLE_EQ(10.5, result.header.timestamp);
  EXPECT_EQ(0U, result.header.proposal_count);
  EXPECT_EQ(9U, result.header.feature_dim);
  EXPECT_TRUE(result.records.empty());
}

TEST(ProposalExportModuleTest, SingleProposalExport) {
  const std::string dir = TestDir("proposal_export_single");
  centerpoint::ProposalExportParam config;
  config.set_enable_proposal_export(true);
  config.set_proposal_export_path(dir);

  ProposalExportModule module;
  ASSERT_TRUE(module.Init(config));
  ASSERT_TRUE(module.Export(20.25, {MakeProposal(2.0f, 0.75f, 3)}));

  const ReadExportFileResult result = ReadExportFile(FirstBinaryPath(dir));
  ASSERT_EQ(1U, result.records.size());
  EXPECT_FLOAT_EQ(2.0f, result.records[0].x);
  EXPECT_FLOAT_EQ(3.0f, result.records[0].y);
  EXPECT_FLOAT_EQ(4.0f, result.records[0].z);
  EXPECT_FLOAT_EQ(5.0f, result.records[0].length);
  EXPECT_FLOAT_EQ(6.0f, result.records[0].width);
  EXPECT_FLOAT_EQ(7.0f, result.records[0].height);
  EXPECT_FLOAT_EQ(8.0f, result.records[0].yaw);
  EXPECT_FLOAT_EQ(0.75f, result.records[0].score);
  EXPECT_EQ(3, result.records[0].class_id);
}

TEST(ProposalExportModuleTest, TopKLimit) {
  const std::string dir = TestDir("proposal_export_top_k");
  centerpoint::ProposalExportParam config;
  config.set_enable_proposal_export(true);
  config.set_proposal_export_path(dir);
  config.set_proposal_export_top_k(2);

  ProposalExportModule module;
  ASSERT_TRUE(module.Init(config));
  ASSERT_TRUE(module.Export(30.0, {
                                      MakeProposal(1.0f, 0.5f, 1),
                                      MakeProposal(2.0f, 0.9f, 2),
                                      MakeProposal(3.0f, 0.8f, 3),
                                  }));

  const ReadExportFileResult result = ReadExportFile(FirstBinaryPath(dir));
  ASSERT_EQ(2U, result.records.size());
  EXPECT_EQ(2, result.records[0].class_id);
  EXPECT_EQ(3, result.records[1].class_id);
}

TEST(ProposalExportModuleTest, MinScoreFilter) {
  const std::string dir = TestDir("proposal_export_min_score");
  centerpoint::ProposalExportParam config;
  config.set_enable_proposal_export(true);
  config.set_proposal_export_path(dir);
  config.set_proposal_export_min_score(0.6f);

  ProposalExportModule module;
  ASSERT_TRUE(module.Init(config));
  ASSERT_TRUE(module.Export(40.0, {
                                      MakeProposal(1.0f, 0.59f, 1),
                                      MakeProposal(2.0f, 0.6f, 2),
                                      MakeProposal(3.0f, 0.9f, 3),
                                  }));

  const ReadExportFileResult result = ReadExportFile(FirstBinaryPath(dir));
  ASSERT_EQ(2U, result.records.size());
  EXPECT_EQ(3, result.records[0].class_id);
  EXPECT_EQ(2, result.records[1].class_id);
}

TEST(ProposalExportModuleTest, FileNamingDoesNotOverwrite) {
  const std::string dir = TestDir("proposal_export_no_overwrite");
  const std::string file_path = FirstBinaryPath(dir);
  {
    std::ofstream existing(file_path, std::ios::out);
    existing << "existing";
  }

  centerpoint::ProposalExportParam config;
  config.set_enable_proposal_export(true);
  config.set_proposal_export_path(dir);

  ProposalExportModule module;
  ASSERT_TRUE(module.Init(config));
  EXPECT_FALSE(module.Export(50.0, {MakeProposal(1.0f, 0.5f, 1)}));
  EXPECT_EQ(0U, module.exported_frame_count());
}

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
