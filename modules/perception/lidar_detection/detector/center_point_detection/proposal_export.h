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

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "modules/perception/lidar_detection/detector/center_point_detection/proposal_refine.h"
#include "modules/perception/lidar_detection/detector/center_point_detection/proto/model_param.pb.h"

namespace apollo {
namespace perception {
namespace lidar {

struct ProposalExportRecord {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float length = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float yaw = 0.0f;
  float score = 0.0f;
  int32_t class_id = 0;
};

struct ProposalExportHeader {
  char magic[8] = {'A', 'P', 'C', 'P', 'R', 'O', 'P', '1'};
  uint32_t version = 1;
  uint32_t record_size = sizeof(ProposalExportRecord);
  uint64_t sequence_id = 0;
  double timestamp = 0.0;
  uint32_t proposal_count = 0;
  uint32_t feature_dim = 9;
};

class ProposalExportModule {
 public:
  bool Init(const centerpoint::ProposalExportParam& config);

  bool enabled() const { return enabled_; }
  bool strict_export() const { return strict_export_; }
  size_t exported_frame_count() const { return exported_frame_count_; }

  bool Export(double timestamp, const std::vector<Proposal>& proposals);

  static std::string BinaryFileName(size_t sequence_id);
  static std::string CsvFileName(size_t sequence_id);

 private:
  std::vector<size_t> SelectProposalIndices(
      const std::vector<Proposal>& proposals) const;
  bool WriteBinaryFile(const std::string& file_path, uint64_t sequence_id,
                       double timestamp, const std::vector<Proposal>& proposals,
                       const std::vector<size_t>& selected_indices) const;
  bool WriteCsvFile(const std::string& file_path, uint64_t sequence_id,
                    double timestamp, const std::vector<Proposal>& proposals,
                    const std::vector<size_t>& selected_indices) const;

  bool enabled_ = false;
  std::string export_path_;
  int top_k_ = -1;
  float min_score_ = 0.0f;
  int max_frames_ = -1;
  bool debug_csv_ = false;
  bool strict_export_ = false;
  size_t exported_frame_count_ = 0;
};

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
