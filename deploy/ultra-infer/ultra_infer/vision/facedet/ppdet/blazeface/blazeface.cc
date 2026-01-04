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

#include "ultra_infer/vision/facedet/ppdet/blazeface/blazeface.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace facedet {

BlazeFace::BlazeFace(const std::string &model_file,
                     const std::string &params_file,
                     const std::string &config_file,
                     const RuntimeOption &custom_option,
                     const ModelFormat &model_format)
    : preprocessor_(config_file) {
  valid_cpu_backends = {Backend::OPENVINO, Backend::PDINFER, Backend::LITE};
  valid_gpu_backends = {Backend::OPENVINO, Backend::LITE, Backend::PDINFER};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool BlazeFace::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

bool BlazeFace::Predict(const cv::Mat &im, FaceDetectionResult *result) {
  std::vector<FaceDetectionResult> results;
  if (!this->BatchPredict({im}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool BlazeFace::BatchPredict(const std::vector<cv::Mat> &images,
                             std::vector<FaceDetectionResult> *results) {
  std::vector<FDMat> fd_images = WrapMat(images);
  FDASSERT(images.size() == 1, "Only support batch = 1 now.");
  std::vector<std::map<std::string, std::array<float, 2>>> ims_info;
  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_, &ims_info)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }

  reused_input_tensors_[0].name = "image";
  reused_input_tensors_[1].name = "scale_factor";
  reused_input_tensors_[2].name = "im_shape";

  // Some models don't need scale_factor and im_shape as input
  while (reused_input_tensors_.size() != NumInputsOfRuntime()) {
    reused_input_tensors_.pop_back();
  }

  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors_, results, ims_info)) {
    FDERROR << "Failed to postprocess the inference results by runtime."
            << std::endl;
    return false;
  }

  return true;
}

} // namespace facedet

} // namespace vision

} // namespace ultra_infer
