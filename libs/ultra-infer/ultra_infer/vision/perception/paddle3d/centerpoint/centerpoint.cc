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

#include "ultra_infer/vision/perception/paddle3d/centerpoint/centerpoint.h"

namespace ultra_infer {
namespace vision {
namespace perception {

Centerpoint::Centerpoint(const std::string &model_file,
                         const std::string &params_file,
                         const std::string &config_file,
                         const RuntimeOption &custom_option,
                         const ModelFormat &model_format)
    : preprocessor_(config_file) {
  valid_gpu_backends = {Backend::PDINFER};

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool Centerpoint::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

bool Centerpoint::Predict(const std::string point_dir,
                          PerceptionResult *result) {
  std::vector<PerceptionResult> results;
  if (!BatchPredict({point_dir}, &results)) {
    return false;
  }

  if (results.size()) {
    *result = std::move(results[0]);
  }
  return true;
}

bool Centerpoint::BatchPredict(std::vector<std::string> points_dir,
                               std::vector<PerceptionResult> *results) {
  int64_t num_point_dim = 5;
  int with_timelag = 0;
  if (!preprocessor_.Run(points_dir, num_point_dim, with_timelag,
                         reused_input_tensors_)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }

  results->resize(reused_input_tensors_.size());
  for (int index = 0; index < reused_input_tensors_.size(); ++index) {
    std::vector<FDTensor> input_tensor;
    input_tensor.push_back(reused_input_tensors_[index]);

    input_tensor[0].name = InputInfoOfRuntime(0).name;

    if (!Infer(input_tensor, &reused_output_tensors_)) {
      FDERROR << "Failed to inference by runtime." << std::endl;
      return false;
    }

    (*results)[index].Clear();
    (*results)[index].Reserve(reused_output_tensors_[0].shape[0]);
    if (!postprocessor_.Run(reused_output_tensors_, &((*results)[index]))) {
      FDERROR << "Failed to postprocess the inference results by runtime."
              << std::endl;
      return false;
    }
  }
  return true;
}

} // namespace perception
} // namespace vision
} // namespace ultra_infer
