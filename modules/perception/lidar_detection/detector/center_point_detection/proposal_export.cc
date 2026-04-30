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

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "cyber/common/file.h"

namespace apollo {
namespace perception {
namespace lidar {
namespace {

std::string JoinPath(const std::string& dir, const std::string& file) {
  if (dir.empty() || dir.back() == '/') {
    return dir + file;
  }
  return dir + "/" + file;
}

template <typename T>
void WriteValue(std::ofstream* out, const T& value) {
  out->write(reinterpret_cast<const char*>(&value), sizeof(T));
}

ProposalExportRecord ToExportRecord(const Proposal& proposal) {
  ProposalExportRecord record;
  record.x = proposal.x;
  record.y = proposal.y;
  record.z = proposal.z;
  record.length = proposal.length;
  record.width = proposal.width;
  record.height = proposal.height;
  record.yaw = proposal.yaw;
  record.score = proposal.score;
  record.class_id = proposal.class_id;
  return record;
}

}  // namespace

bool ProposalExportModule::Init(
    const centerpoint::ProposalExportParam& config) {
  enabled_ = config.enable_proposal_export();
  export_path_ = config.proposal_export_path();
  top_k_ = config.proposal_export_top_k();
  min_score_ = config.proposal_export_min_score();
  max_frames_ = config.proposal_export_max_frames();
  debug_csv_ = config.proposal_export_debug_csv();
  strict_export_ = config.strict_export();
  exported_frame_count_ = 0;

  if (!enabled_) {
    return true;
  }
  return !export_path_.empty() && top_k_ >= -1 && max_frames_ >= -1;
}

bool ProposalExportModule::Export(double timestamp,
                                  const std::vector<Proposal>& proposals) {
  if (!enabled_) {
    return true;
  }
  if (max_frames_ >= 0 &&
      exported_frame_count_ >= static_cast<size_t>(max_frames_)) {
    return true;
  }
  if (!apollo::cyber::common::EnsureDirectory(export_path_)) {
    return false;
  }

  const uint64_t sequence_id = exported_frame_count_;
  const std::string binary_path =
      JoinPath(export_path_, BinaryFileName(sequence_id));
  if (apollo::cyber::common::PathExists(binary_path)) {
    return false;
  }
  const std::string csv_path = JoinPath(export_path_, CsvFileName(sequence_id));
  if (debug_csv_ && apollo::cyber::common::PathExists(csv_path)) {
    return false;
  }
  const std::vector<size_t> selected_indices = SelectProposalIndices(proposals);
  if (!WriteBinaryFile(binary_path, sequence_id, timestamp, proposals,
                       selected_indices)) {
    return false;
  }

  if (debug_csv_) {
    if (!WriteCsvFile(csv_path, sequence_id, timestamp, proposals,
                      selected_indices)) {
      return false;
    }
  }
  ++exported_frame_count_;
  return true;
}

std::string ProposalExportModule::BinaryFileName(size_t sequence_id) {
  std::ostringstream oss;
  oss << "proposal_" << std::setw(6) << std::setfill('0') << sequence_id
      << ".bin";
  return oss.str();
}

std::string ProposalExportModule::CsvFileName(size_t sequence_id) {
  std::ostringstream oss;
  oss << "proposal_" << std::setw(6) << std::setfill('0') << sequence_id
      << ".csv";
  return oss.str();
}

std::vector<size_t> ProposalExportModule::SelectProposalIndices(
    const std::vector<Proposal>& proposals) const {
  std::vector<size_t> selected_indices;
  selected_indices.reserve(proposals.size());
  for (size_t i = 0; i < proposals.size(); ++i) {
    if (proposals[i].score >= min_score_) {
      selected_indices.push_back(i);
    }
  }

  std::stable_sort(selected_indices.begin(), selected_indices.end(),
                   [&proposals](size_t lhs, size_t rhs) {
                     return proposals[lhs].score > proposals[rhs].score;
                   });
  if (top_k_ >= 0 && selected_indices.size() > static_cast<size_t>(top_k_)) {
    selected_indices.resize(static_cast<size_t>(top_k_));
  }
  return selected_indices;
}

bool ProposalExportModule::WriteBinaryFile(
    const std::string& file_path, uint64_t sequence_id, double timestamp,
    const std::vector<Proposal>& proposals,
    const std::vector<size_t>& selected_indices) const {
  std::ofstream out(file_path, std::ios::binary | std::ios::out);
  if (!out.is_open()) {
    return false;
  }

  ProposalExportHeader header;
  header.sequence_id = sequence_id;
  header.timestamp = timestamp;
  header.proposal_count = static_cast<uint32_t>(selected_indices.size());
  out.write(header.magic, sizeof(header.magic));
  WriteValue(&out, header.version);
  WriteValue(&out, header.record_size);
  WriteValue(&out, header.sequence_id);
  WriteValue(&out, header.timestamp);
  WriteValue(&out, header.proposal_count);
  WriteValue(&out, header.feature_dim);

  for (const size_t index : selected_indices) {
    const ProposalExportRecord record = ToExportRecord(proposals[index]);
    WriteValue(&out, record.x);
    WriteValue(&out, record.y);
    WriteValue(&out, record.z);
    WriteValue(&out, record.length);
    WriteValue(&out, record.width);
    WriteValue(&out, record.height);
    WriteValue(&out, record.yaw);
    WriteValue(&out, record.score);
    WriteValue(&out, record.class_id);
  }
  return out.good();
}

bool ProposalExportModule::WriteCsvFile(
    const std::string& file_path, uint64_t sequence_id, double timestamp,
    const std::vector<Proposal>& proposals,
    const std::vector<size_t>& selected_indices) const {
  std::ofstream out(file_path, std::ios::out);
  if (!out.is_open()) {
    return false;
  }
  out << "sequence_id,timestamp,x,y,z,length,width,height,yaw,score,class_id\n";
  for (const size_t index : selected_indices) {
    const Proposal& proposal = proposals[index];
    out << sequence_id << "," << timestamp << "," << proposal.x << ","
        << proposal.y << "," << proposal.z << "," << proposal.length << ","
        << proposal.width << "," << proposal.height << "," << proposal.yaw
        << "," << proposal.score << "," << proposal.class_id << "\n";
  }
  return out.good();
}

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
