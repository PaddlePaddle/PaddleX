// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "ultra_infer/vision/ocr/ppocr/structurev2_layout.h"

#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

StructureV2Layout::StructureV2Layout() {}
StructureV2Layout::StructureV2Layout(const std::string &model_file,
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

bool StructureV2Layout::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

std::unique_ptr<StructureV2Layout> StructureV2Layout::Clone() const {
  std::unique_ptr<StructureV2Layout> clone_model =
      utils::make_unique<StructureV2Layout>(StructureV2Layout(*this));
  clone_model->SetRuntime(clone_model->CloneRuntime());
  return clone_model;
}

bool StructureV2Layout::Predict(cv::Mat *im, DetectionResult *result) {
  return Predict(*im, result);
}

bool StructureV2Layout::Predict(const cv::Mat &im, DetectionResult *result) {
  std::vector<DetectionResult> results;
  if (!BatchPredict({im}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool StructureV2Layout::BatchPredict(const std::vector<cv::Mat> &images,
                                     std::vector<DetectionResult> *results) {
  std::vector<FDMat> fd_images = WrapMat(images);
  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }
  auto batch_layout_img_info = preprocessor_.GetBatchLayoutImgInfo();

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors_, results,
                          *batch_layout_img_info)) {
    FDERROR << "Failed to postprocess the inference results." << std::endl;
    return false;
  }
  return true;
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
