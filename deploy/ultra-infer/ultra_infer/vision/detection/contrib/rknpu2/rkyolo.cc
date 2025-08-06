// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.  //NOLINT
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
#include "ultra_infer/vision/detection/contrib/rknpu2/rkyolo.h"

namespace ultra_infer {
namespace vision {
namespace detection {

RKYOLO::RKYOLO(const std::string &model_file,
               const ultra_infer::RuntimeOption &custom_option,
               const ultra_infer::ModelFormat &model_format) {
  if (model_format == ModelFormat::RKNN) {
    valid_cpu_backends = {};
    valid_gpu_backends = {};
    valid_rknpu_backends = {Backend::RKNPU2};
  } else {
    FDERROR << "RKYOLO Only Support run in RKNPU2" << std::endl;
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  initialized = Initialize();
}

bool RKYOLO::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  auto size = GetPreprocessor().GetSize();
  GetPostprocessor().SetHeightAndWeight(size[0], size[1]);
  return true;
}

bool RKYOLO::Predict(const cv::Mat &im, DetectionResult *result) {
  std::vector<DetectionResult> results;
  if (!BatchPredict({im}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool RKYOLO::BatchPredict(const std::vector<cv::Mat> &images,
                          std::vector<DetectionResult> *results) {
  std::vector<FDMat> fd_images = WrapMat(images);

  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  auto pad_hw_values_ = preprocessor_.GetPadHWValues();
  postprocessor_.SetPadHWValues(preprocessor_.GetPadHWValues());
  postprocessor_.SetScale(preprocessor_.GetScale());
  if (!postprocessor_.Run(reused_output_tensors_, results)) {
    FDERROR << "Failed to postprocess the inference results by runtime."
            << std::endl;
    return false;
  }
  return true;
}

} // namespace detection
} // namespace vision
} // namespace ultra_infer
