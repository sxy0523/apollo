/******************************************************************************
 * Copyright 2024 The Apollo Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <vector>

#include "modules/perception/lidar/lib/interface/base_point_cloud_obstacle_detection.h"
#include "modules/perception/lidar/lib/quantum_detection/quantum_layer.h"

namespace apollo {
namespace perception {
namespace lidar {

struct QuantumPointPillarsOptions {
    // Model configuration paths
    std::string pillar_scatter_model_path;
    std::string backbone_model_path;
    std::string detection_head_model_path;

    // Quantum computing settings
    int num_qubits = 4;
    int num_layers = 2;
    bool use_hybrid_mode = true;

    // Inference settings
    int max_num_points = 16384;
    int max_num_pillars = 12000;
    int num_classes = 3;

    // Detection range [min_x, max_x, min_y, max_y, min_z, max_z]
    std::vector<float> point_cloud_range = {-69.12f, 69.12f, -69.12f, 69.12f, -3.0f, 5.0f};

    // Pillar settings
    float pillar_x_size = 0.16f;
    float pillar_y_size = 0.16f;
    float pillar_z_size = 0.2f;
};

class QuantumPointPillars : public BasePointCloudObstacleDetection {
public:
    QuantumPointPillars() = default;
    ~QuantumPointPillars() = default;

    bool Init(const PointCloudObstacleDetectionInitOptions& options) override;
    bool Detect(const PointCloudDetectionOptions& options,
                LidarFrame* frame) override;
    bool InitWithConfig(const PointCloudObstacleDetectionInitOptions& options);

private:
    bool ProcessPointCloud(const PointCloud& point_cloud,
                          std::vector<ObjectPtr>* objects);

    // Quantum-enhanced feature extraction
    bool ExtractQuantumFeatures(const PointCloud& point_cloud,
                               Eigen::MatrixXf* pillar_features);

    // Classical post-processing
    bool PostProcess(const Eigen::MatrixXf& detections,
                    std::vector<ObjectPtr>* objects);

    // Convert point cloud to pillar representation
    bool ConvertToPillars(const PointCloud& point_cloud,
                          Eigen::MatrixXf* pillars,
                          Eigen::MatrixXi* pillar_indices);

    // Initialize quantum backend
    bool InitQuantumBackend();

    // Shared quantum layer for feature extraction
    std::unique_ptr<QuantumLayer> quantum_layer_;

    // Model parameters
    QuantumPointPillarsOptions options_;

    // Intermediate buffers
    Eigen::MatrixXf pillar_features_;
    Eigen::MatrixXf scattered_features_;
    Eigen::MatrixXf backbone_output_;
};

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
