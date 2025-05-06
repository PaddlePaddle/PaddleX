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

#include "ultra_infer/vision/perception/paddle3d/petr/petr.h"

namespace ultra_infer {
namespace vision {
namespace perception {

Petr::Petr(const std::string &model_file, const std::string &params_file,
           const std::string &config_file, const RuntimeOption &custom_option,
           const ModelFormat &model_format)
    : preprocessor_(config_file) {
  valid_cpu_backends = {Backend::PDINFER};
  valid_gpu_backends = {Backend::PDINFER};

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  runtime_option.paddle_infer_option.enable_mkldnn = false;
  initialized = Initialize();
}

bool Petr::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

bool Petr::Predict(const cv::Mat &images, PerceptionResult *results) {
  FDERROR << "Petr inference only support 6(V1) or 12(V2) images" << std::endl;
  return false;
}

bool Petr::BatchPredict(const std::vector<cv::Mat> &images,
                        std::vector<PerceptionResult> *results) {
  if ((images.size() != 6) && (images.size() != 12)) {
    FDERROR << "Petr only support 6(V1) or 12(V2) images";
    return false;
  }
  std::vector<FDMat> fd_images = WrapMat(images);

  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }

  // Note: un-commented the codes below to show the debug info.
  // reused_input_tensors_[0].PrintInfo();
  // reused_input_tensors_[1].PrintInfo();
  // reused_input_tensors_[2].PrintInfo();

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  reused_input_tensors_[1].name = InputInfoOfRuntime(1).name;
  if (images.size() == 12) {
    // for Petr V2 timestamp
    reused_input_tensors_[2].name = InputInfoOfRuntime(2).name;
  } else {
    // for Petr V1
    reused_input_tensors_.pop_back();
  }

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
