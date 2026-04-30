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

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>

#include "cyber/common/file.h"
#include "cyber/common/log.h"
#include "modules/common/math/math_utils.h"

namespace apollo {
namespace perception {
namespace lidar {
namespace {

constexpr char kMockZeroMode[] = "mock_zero";
constexpr char kMockDeltaMode[] = "mock_delta";
constexpr char kStudentModelMode[] = "student_model";
constexpr float kMinBoxDimension = 1.0e-3f;

const char* const kExpectedFeatureOrder[] = {
    "x", "y", "z", "length", "width", "height", "yaw", "score", "class_id",
};

const char* const kExpectedDeltaOrder[] = {
    "delta_x",     "delta_y",      "delta_z",   "delta_length",
    "delta_width", "delta_height", "delta_yaw", "delta_score",
};

struct OnnxTensor {
  std::string name;
  std::vector<int64_t> dims;
  std::vector<float> values;
};

struct StudentLayer {
  int input_dim = 0;
  int output_dim = 0;
  std::vector<float> weights;
  std::vector<float> bias;
};

std::vector<std::string> ExpectedFeatureOrder() {
  return std::vector<std::string>(
      kExpectedFeatureOrder,
      kExpectedFeatureOrder +
          sizeof(kExpectedFeatureOrder) / sizeof(kExpectedFeatureOrder[0]));
}

std::vector<std::string> ExpectedDeltaOrder() {
  return std::vector<std::string>(
      kExpectedDeltaOrder,
      kExpectedDeltaOrder +
          sizeof(kExpectedDeltaOrder) / sizeof(kExpectedDeltaOrder[0]));
}

bool ReadVarint(const std::string& data, size_t* offset, uint64_t* value) {
  if (offset == nullptr || value == nullptr) {
    return false;
  }
  uint64_t result = 0;
  int shift = 0;
  while (*offset < data.size() && shift <= 63) {
    const uint8_t byte = static_cast<uint8_t>(data[*offset]);
    ++(*offset);
    result |= static_cast<uint64_t>(byte & 0x7f) << shift;
    if ((byte & 0x80) == 0) {
      *value = result;
      return true;
    }
    shift += 7;
  }
  return false;
}

bool ReadNextProtoField(const std::string& data, size_t* offset,
                        int* field_number, int* wire_type, uint64_t* varint,
                        std::string* payload) {
  uint64_t key = 0;
  if (!ReadVarint(data, offset, &key)) {
    return false;
  }
  *field_number = static_cast<int>(key >> 3);
  *wire_type = static_cast<int>(key & 0x07);
  *varint = 0;
  payload->clear();
  if (*wire_type == 0) {
    return ReadVarint(data, offset, varint);
  }
  if (*wire_type == 2) {
    uint64_t length = 0;
    if (!ReadVarint(data, offset, &length) || *offset + length > data.size()) {
      return false;
    }
    *payload = data.substr(*offset, static_cast<size_t>(length));
    *offset += static_cast<size_t>(length);
    return true;
  }
  return false;
}

bool ParseOnnxTensor(const std::string& data, OnnxTensor* tensor,
                     std::string* error) {
  size_t offset = 0;
  int data_type = 0;
  std::string raw_data;
  while (offset < data.size()) {
    int field_number = 0;
    int wire_type = 0;
    uint64_t varint = 0;
    std::string payload;
    if (!ReadNextProtoField(data, &offset, &field_number, &wire_type, &varint,
                            &payload)) {
      *error = "failed to parse ONNX TensorProto";
      return false;
    }
    if (field_number == 1 && wire_type == 0) {
      tensor->dims.push_back(static_cast<int64_t>(varint));
    } else if (field_number == 2 && wire_type == 0) {
      data_type = static_cast<int>(varint);
    } else if (field_number == 8 && wire_type == 2) {
      tensor->name = payload;
    } else if (field_number == 9 && wire_type == 2) {
      raw_data = payload;
    }
  }

  if (tensor->name.empty() || tensor->dims.empty() || data_type != 1) {
    *error = "unsupported ONNX TensorProto initializer";
    return false;
  }
  int64_t element_count = 1;
  for (const int64_t dim : tensor->dims) {
    if (dim <= 0) {
      *error = "invalid ONNX tensor dimension";
      return false;
    }
    element_count *= dim;
  }
  if (raw_data.size() != static_cast<size_t>(element_count) * sizeof(float)) {
    *error = "ONNX tensor raw_data size mismatch";
    return false;
  }
  tensor->values.resize(static_cast<size_t>(element_count));
  std::memcpy(tensor->values.data(), raw_data.data(), raw_data.size());
  return true;
}

bool LoadOnnxInitializers(const std::string& model_path,
                          std::map<std::string, OnnxTensor>* initializers,
                          std::string* error) {
  std::string model_data;
  if (!apollo::cyber::common::GetContent(model_path, &model_data)) {
    *error = "failed to read student model file: " + model_path;
    return false;
  }

  std::string graph_data;
  size_t offset = 0;
  while (offset < model_data.size()) {
    int field_number = 0;
    int wire_type = 0;
    uint64_t varint = 0;
    std::string payload;
    if (!ReadNextProtoField(model_data, &offset, &field_number, &wire_type,
                            &varint, &payload)) {
      *error = "failed to parse ONNX ModelProto";
      return false;
    }
    if (field_number == 7 && wire_type == 2) {
      graph_data = payload;
      break;
    }
  }
  if (graph_data.empty()) {
    *error = "ONNX graph is missing";
    return false;
  }

  offset = 0;
  while (offset < graph_data.size()) {
    int field_number = 0;
    int wire_type = 0;
    uint64_t varint = 0;
    std::string payload;
    if (!ReadNextProtoField(graph_data, &offset, &field_number, &wire_type,
                            &varint, &payload)) {
      *error = "failed to parse ONNX GraphProto";
      return false;
    }
    if (field_number == 5 && wire_type == 2) {
      OnnxTensor tensor;
      if (!ParseOnnxTensor(payload, &tensor, error)) {
        return false;
      }
      (*initializers)[tensor.name] = tensor;
    }
  }
  if (initializers->empty()) {
    *error = "ONNX graph has no initializers";
    return false;
  }
  return true;
}

std::string FieldKey(const std::string& name) { return "\"" + name + "\""; }

bool ExtractArrayPayload(const std::string& json, const std::string& field_name,
                         std::string* payload) {
  const size_t key_pos = json.find(FieldKey(field_name));
  if (key_pos == std::string::npos) {
    return false;
  }
  const size_t begin = json.find('[', key_pos);
  if (begin == std::string::npos) {
    return false;
  }
  int depth = 0;
  for (size_t i = begin; i < json.size(); ++i) {
    if (json[i] == '[') {
      ++depth;
    } else if (json[i] == ']') {
      --depth;
      if (depth == 0) {
        *payload = json.substr(begin + 1, i - begin - 1);
        return true;
      }
    }
  }
  return false;
}

bool ExtractIntField(const std::string& json, const std::string& field_name,
                     int* value) {
  const size_t key_pos = json.find(FieldKey(field_name));
  if (key_pos == std::string::npos) {
    return false;
  }
  const size_t colon = json.find(':', key_pos);
  if (colon == std::string::npos) {
    return false;
  }
  const char* start = json.c_str() + colon + 1;
  char* end = nullptr;
  const long parsed = std::strtol(start, &end, 10);
  if (end == start) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool ParseStringArray(const std::string& payload,
                      std::vector<std::string>* values) {
  values->clear();
  size_t pos = 0;
  while (pos < payload.size()) {
    const size_t begin = payload.find('"', pos);
    if (begin == std::string::npos) {
      break;
    }
    const size_t end = payload.find('"', begin + 1);
    if (end == std::string::npos) {
      return false;
    }
    values->push_back(payload.substr(begin + 1, end - begin - 1));
    pos = end + 1;
  }
  return !values->empty();
}

bool ParseFloatArray(const std::string& payload, std::vector<float>* values) {
  values->clear();
  size_t pos = 0;
  while (pos < payload.size()) {
    while (pos < payload.size() &&
           (payload[pos] == ',' || std::isspace(payload[pos]))) {
      ++pos;
    }
    if (pos >= payload.size()) {
      break;
    }
    const char* start = payload.c_str() + pos;
    char* end = nullptr;
    const float parsed = std::strtof(start, &end);
    if (end == start) {
      return false;
    }
    values->push_back(parsed);
    pos = static_cast<size_t>(end - payload.c_str());
  }
  return !values->empty();
}

bool ExtractStringArrayField(const std::string& json,
                             const std::string& field_name,
                             std::vector<std::string>* values) {
  std::string payload;
  return ExtractArrayPayload(json, field_name, &payload) &&
         ParseStringArray(payload, values);
}

bool ExtractFloatArrayField(const std::string& json,
                            const std::string& field_name,
                            std::vector<float>* values) {
  std::string payload;
  return ExtractArrayPayload(json, field_name, &payload) &&
         ParseFloatArray(payload, values);
}

}  // namespace

struct ProposalRefineModule::StudentModel {
  int input_dim = 0;
  int output_dim = 0;
  std::vector<float> mean;
  std::vector<float> stddev;
  std::vector<StudentLayer> layers;

  bool Load(const std::string& model_path, const std::string& metadata_path,
            std::string* error) {
    if (!LoadMetadata(metadata_path, error)) {
      return false;
    }
    return LoadModel(model_path, error);
  }

  bool LoadMetadata(const std::string& metadata_path, std::string* error) {
    std::string json;
    if (!apollo::cyber::common::GetContent(metadata_path, &json)) {
      *error = "failed to read student metadata file: " + metadata_path;
      return false;
    }
    std::vector<std::string> feature_order;
    std::vector<std::string> delta_order;
    if (!ExtractIntField(json, "input_dim", &input_dim) ||
        !ExtractIntField(json, "output_dim", &output_dim) ||
        !ExtractStringArrayField(json, "feature_order", &feature_order) ||
        !ExtractStringArrayField(json, "delta_order", &delta_order) ||
        !ExtractFloatArrayField(json, "mean", &mean) ||
        !ExtractFloatArrayField(json, "std", &stddev)) {
      *error = "student metadata is missing required fields";
      return false;
    }

    if (input_dim != static_cast<int>(ExpectedFeatureOrder().size()) ||
        output_dim != static_cast<int>(ExpectedDeltaOrder().size()) ||
        feature_order != ExpectedFeatureOrder() ||
        delta_order != ExpectedDeltaOrder() ||
        mean.size() != static_cast<size_t>(input_dim) ||
        stddev.size() != static_cast<size_t>(input_dim)) {
      *error = "student metadata dimensions or feature order mismatch";
      return false;
    }
    for (size_t i = 0; i < stddev.size(); ++i) {
      if (!std::isfinite(mean[i]) || !std::isfinite(stddev[i]) ||
          stddev[i] <= 0.0f) {
        *error = "student metadata normalization values are invalid";
        return false;
      }
    }
    return true;
  }

  bool LoadModel(const std::string& model_path, std::string* error) {
    std::map<std::string, OnnxTensor> initializers;
    if (!LoadOnnxInitializers(model_path, &initializers, error)) {
      return false;
    }

    layers.clear();
    int expected_input_dim = input_dim;
    for (int layer_index = 0;; ++layer_index) {
      std::ostringstream weight_name;
      std::ostringstream bias_name;
      weight_name << "weight_" << layer_index;
      bias_name << "bias_" << layer_index;
      const auto weight_iter = initializers.find(weight_name.str());
      const auto bias_iter = initializers.find(bias_name.str());
      if (weight_iter == initializers.end() ||
          bias_iter == initializers.end()) {
        break;
      }
      const OnnxTensor& weight = weight_iter->second;
      const OnnxTensor& bias = bias_iter->second;
      if (weight.dims.size() != 2 || bias.dims.size() != 1 ||
          weight.dims[0] != expected_input_dim ||
          weight.dims[1] != bias.dims[0]) {
        *error = "student ONNX layer dimensions are invalid";
        return false;
      }
      StudentLayer layer;
      layer.input_dim = static_cast<int>(weight.dims[0]);
      layer.output_dim = static_cast<int>(weight.dims[1]);
      layer.weights = weight.values;
      layer.bias = bias.values;
      layers.push_back(layer);
      expected_input_dim = layer.output_dim;
    }
    if (layers.empty() || layers.back().output_dim != output_dim) {
      *error = "student ONNX output dimension mismatch";
      return false;
    }
    return true;
  }

  bool BuildFeatureBatch(const std::vector<Proposal>& proposals,
                         const std::vector<size_t>& selected_indices,
                         std::vector<float>* features) const {
    features->assign(selected_indices.size() * input_dim, 0.0f);
    for (size_t row = 0; row < selected_indices.size(); ++row) {
      const Proposal& proposal = proposals[selected_indices[row]];
      const float raw_values[] = {
          proposal.x,
          proposal.y,
          proposal.z,
          proposal.length,
          proposal.width,
          proposal.height,
          proposal.yaw,
          proposal.score,
          static_cast<float>(proposal.class_id),
      };
      for (int col = 0; col < input_dim; ++col) {
        (*features)[row * input_dim + col] =
            (raw_values[col] - mean[col]) / stddev[col];
      }
    }
    return true;
  }

  bool Predict(const std::vector<float>& features,
               std::vector<ProposalDelta>* deltas) const {
    if (input_dim <= 0 || features.size() % input_dim != 0) {
      return false;
    }
    const size_t batch_size = features.size() / input_dim;
    deltas->clear();
    deltas->reserve(batch_size);
    for (size_t row = 0; row < batch_size; ++row) {
      std::vector<float> values(features.begin() + row * input_dim,
                                features.begin() + (row + 1) * input_dim);
      for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const StudentLayer& layer = layers[layer_index];
        std::vector<float> next(layer.output_dim, 0.0f);
        for (int out = 0; out < layer.output_dim; ++out) {
          float value = layer.bias[out];
          for (int in = 0; in < layer.input_dim; ++in) {
            value += values[in] * layer.weights[in * layer.output_dim + out];
          }
          if (layer_index + 1 < layers.size()) {
            value = std::max(0.0f, value);
          }
          next[out] = value;
        }
        values.swap(next);
      }
      if (values.size() != static_cast<size_t>(output_dim)) {
        return false;
      }
      ProposalDelta delta;
      delta.delta_x = values[0];
      delta.delta_y = values[1];
      delta.delta_z = values[2];
      delta.delta_length = values[3];
      delta.delta_width = values[4];
      delta.delta_height = values[5];
      delta.delta_yaw = values[6];
      delta.delta_score = values[7];
      deltas->push_back(delta);
    }
    return true;
  }
};

ProposalRefineModule::~ProposalRefineModule() { delete student_model_; }

bool ProposalRefineModule::Init(
    const centerpoint::ProposalRefineParam& config) {
  enabled_ = config.enable_proposal_refine();
  mode_ = config.proposal_refine_mode();
  if (mode_.empty()) {
    mode_ = kMockZeroMode;
  }
  top_k_ = config.proposal_refine_top_k();
  score_threshold_ = config.proposal_refine_score_threshold();
  delta_scale_ = config.proposal_refine_delta_scale();
  debug_log_ = config.proposal_refine_debug_log();
  strict_refine_ = config.strict_refine();
  refine_call_count_ = 0;
  delete student_model_;
  student_model_ = nullptr;

  if (!enabled_) {
    return true;
  }
  if (top_k_ < -1) {
    return false;
  }
  if (mode_ == kStudentModelMode) {
    return InitStudentModel(config);
  }
  return mode_ == kMockZeroMode || mode_ == kMockDeltaMode;
}

bool ProposalRefineModule::Refine(std::vector<Proposal>* proposals,
                                  ProposalRefineStats* stats) {
  if (proposals == nullptr || stats == nullptr) {
    return false;
  }
  ++refine_call_count_;

  const auto start = std::chrono::steady_clock::now();
  stats->proposal_count = proposals->size();
  const std::vector<size_t> selected_indices =
      SelectProposalIndices(*proposals);
  stats->refined_proposal_count = selected_indices.size();
  stats->delta_min = 0.0f;
  stats->delta_max = 0.0f;
  stats->feature_build_latency_ms = 0.0;
  stats->inference_latency_ms = 0.0;
  stats->delta_apply_latency_ms = 0.0;
  if (mode_ == kMockDeltaMode && !selected_indices.empty()) {
    const ProposalDelta delta = GetMockDelta();
    stats->delta_min = std::numeric_limits<float>::max();
    stats->delta_max = std::numeric_limits<float>::lowest();
    const float raw_deltas[] = {
        delta.delta_x,      delta.delta_y,     delta.delta_z,
        delta.delta_length, delta.delta_width, delta.delta_height,
        delta.delta_yaw,    delta.delta_score,
    };
    for (const float value : raw_deltas) {
      stats->delta_min = std::min(stats->delta_min, value);
      stats->delta_max = std::max(stats->delta_max, value);
    }
    for (const size_t index : selected_indices) {
      ApplyProposalDelta(delta, delta_scale_, &proposals->at(index));
    }
  } else if (mode_ == kStudentModelMode && !selected_indices.empty()) {
    if (student_model_ == nullptr) {
      return false;
    }
    const auto feature_start = std::chrono::steady_clock::now();
    std::vector<float> feature_batch;
    if (!student_model_->BuildFeatureBatch(*proposals, selected_indices,
                                           &feature_batch)) {
      return false;
    }
    const auto feature_end = std::chrono::steady_clock::now();
    stats->feature_build_latency_ms =
        std::chrono::duration<double, std::milli>(feature_end - feature_start)
            .count();

    const auto inference_start = std::chrono::steady_clock::now();
    std::vector<ProposalDelta> deltas;
    if (!student_model_->Predict(feature_batch, &deltas) ||
        deltas.size() != selected_indices.size()) {
      return false;
    }
    const auto inference_end = std::chrono::steady_clock::now();
    stats->inference_latency_ms = std::chrono::duration<double, std::milli>(
                                      inference_end - inference_start)
                                      .count();

    const auto delta_start = std::chrono::steady_clock::now();
    stats->delta_min = std::numeric_limits<float>::max();
    stats->delta_max = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < selected_indices.size(); ++i) {
      const ProposalDelta& delta = deltas[i];
      const float raw_deltas[] = {
          delta.delta_x,      delta.delta_y,     delta.delta_z,
          delta.delta_length, delta.delta_width, delta.delta_height,
          delta.delta_yaw,    delta.delta_score,
      };
      for (const float value : raw_deltas) {
        stats->delta_min = std::min(stats->delta_min, value);
        stats->delta_max = std::max(stats->delta_max, value);
      }
      ApplyProposalDelta(delta, delta_scale_,
                         &proposals->at(selected_indices[i]));
    }
    const auto delta_end = std::chrono::steady_clock::now();
    stats->delta_apply_latency_ms =
        std::chrono::duration<double, std::milli>(delta_end - delta_start)
            .count();
  }
  const auto end = std::chrono::steady_clock::now();
  stats->latency_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  return true;
}

std::vector<size_t> ProposalRefineModule::SelectProposalIndices(
    const std::vector<Proposal>& proposals) const {
  std::vector<size_t> selected_indices;
  selected_indices.reserve(proposals.size());
  for (size_t i = 0; i < proposals.size(); ++i) {
    if (proposals[i].score >= score_threshold_) {
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

bool ProposalRefineModule::InitStudentModel(
    const centerpoint::ProposalRefineParam& config) {
  student_model_ = new StudentModel();
  std::string error;
  if (!student_model_->LoadMetadata(config.proposal_refine_metadata_path(),
                                    &error)) {
    AERROR << "Failed to load student proposal refine metadata: " << error;
    delete student_model_;
    student_model_ = nullptr;
    return false;
  }
  if (!student_model_->LoadModel(config.proposal_refine_model_path(), &error)) {
    return DisableStudentModelAfterLoadFailure(error);
  }
  return true;
}

bool ProposalRefineModule::DisableStudentModelAfterLoadFailure(
    const std::string& reason) {
  AERROR << "Failed to load student proposal refine model: " << reason;
  delete student_model_;
  student_model_ = nullptr;
  if (strict_refine_) {
    return false;
  }
  AWARN << "Disabling proposal refine because strict_refine=false.";
  enabled_ = false;
  return true;
}

ProposalDelta GetMockDelta() {
  ProposalDelta delta;
  delta.delta_x = 0.1f;
  delta.delta_yaw = 0.05f;
  delta.delta_score = 0.01f;
  return delta;
}

void ApplyProposalDelta(const ProposalDelta& delta, float delta_scale,
                        Proposal* proposal) {
  if (proposal == nullptr) {
    return;
  }
  proposal->x += delta_scale * delta.delta_x;
  proposal->y += delta_scale * delta.delta_y;
  proposal->z += delta_scale * delta.delta_z;
  proposal->length = std::max(
      kMinBoxDimension, proposal->length + delta_scale * delta.delta_length);
  proposal->width = std::max(kMinBoxDimension,
                             proposal->width + delta_scale * delta.delta_width);
  proposal->height = std::max(
      kMinBoxDimension, proposal->height + delta_scale * delta.delta_height);
  proposal->yaw = static_cast<float>(apollo::common::math::NormalizeAngle(
      proposal->yaw + delta_scale * delta.delta_yaw));
  proposal->score = apollo::common::math::Clamp(
      proposal->score + delta_scale * delta.delta_score, 0.0f, 1.0f);
}

bool MaybeRefineProposals(ProposalRefineModule* module,
                          std::vector<Proposal>* proposals,
                          ProposalRefineStats* stats) {
  if (module == nullptr) {
    return false;
  }
  if (!module->enabled()) {
    if (stats != nullptr) {
      *stats = ProposalRefineStats();
    }
    return true;
  }
  return module->Refine(proposals, stats);
}

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
