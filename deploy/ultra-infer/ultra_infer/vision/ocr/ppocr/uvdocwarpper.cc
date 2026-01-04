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

#include "ultra_infer/vision/ocr/ppocr/uvdocwarpper.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

UVDocWarpper::UVDocWarpper() {}
UVDocWarpper::UVDocWarpper(const std::string &model_file,
                           const std::string &params_file,
                           const RuntimeOption &custom_option,
                           const ModelFormat &model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT, Backend::OPENVINO};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::OPENVINO,
                          Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    valid_sophgonpu_backends = {Backend::SOPHGOTPU};
    valid_rknpu_backends = {Backend::RKNPU2};
  }

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

// Init
bool UVDocWarpper::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

std::unique_ptr<UVDocWarpper> UVDocWarpper::Clone() const {
  std::unique_ptr<UVDocWarpper> clone_model =
      utils::make_unique<UVDocWarpper>(UVDocWarpper(*this));
  clone_model->SetRuntime(clone_model->CloneRuntime());
  return clone_model;
}

bool UVDocWarpper::Predict(cv::Mat *im, FDTensor *result) {
  return Predict(*im, result);
}

bool UVDocWarpper::Predict(const cv::Mat &img, FDTensor *result) {
  std::vector<FDTensor> results;
  if (!BatchPredict({img}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool UVDocWarpper::BatchPredict(const std::vector<cv::Mat> &images,
                                std::vector<FDTensor> *results) {
  std::vector<FDMat> fd_images = WrapMat(images);
  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors_, results)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
