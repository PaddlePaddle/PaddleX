// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ultra_infer/vision/perception/paddle3d/caddn/caddn.h"

namespace ultra_infer {
namespace vision {
namespace perception {

Caddn::Caddn(const std::string &model_file, const std::string &params_file,
             const std::string &config_file, const RuntimeOption &custom_option,
             const ModelFormat &model_format)
    : preprocessor_(config_file) {
  valid_gpu_backends = {Backend::PDINFER};

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool Caddn::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

bool Caddn::Predict(const cv::Mat &im, std::vector<float> &input_cam_data,
                    std::vector<float> &input_lidar_data,
                    PerceptionResult *result) {
  std::vector<PerceptionResult> results;
  if (!BatchPredict({im}, input_cam_data, input_lidar_data, &results)) {
    return false;
  }
  if (results.size()) {
    *result = std::move(results[0]);
  }
  return true;
}

bool Caddn::BatchPredict(const std::vector<cv::Mat> &images,
                         std::vector<float> &input_cam_data,
                         std::vector<float> &input_lidar_data,
                         std::vector<PerceptionResult> *results) {
  std::vector<FDMat> fd_images = WrapMat(images);

  if (!preprocessor_.Run(&fd_images, input_cam_data, input_lidar_data,
                         &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }

  reused_input_tensors_[0].name = "images";
  reused_input_tensors_[1].name = "trans_cam_to_img";
  reused_input_tensors_[2].name = "trans_lidar_to_cam";

  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors_, results)) {
    FDERROR << "Failed to postprocess the inference results by runtime."
            << std::endl;
    return false;
  }
  return true;
}

} // namespace perception
} // namespace vision
} // namespace ultra_infer
