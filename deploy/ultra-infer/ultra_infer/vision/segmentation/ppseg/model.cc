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

#include "ultra_infer/vision/segmentation/ppseg/model.h"

#include "ultra_infer/utils/unique_ptr.h"

namespace ultra_infer {
namespace vision {
namespace segmentation {

PaddleSegModel::PaddleSegModel(const std::string &model_file,
                               const std::string &params_file,
                               const std::string &config_file,
                               const RuntimeOption &custom_option,
                               const ModelFormat &model_format)
    : preprocessor_(config_file), postprocessor_(config_file) {
  if (model_format == ModelFormat::SOPHGO) {
    valid_sophgonpu_backends = {Backend::SOPHGOTPU};
  } else {
    valid_cpu_backends = {Backend::OPENVINO, Backend::PDINFER, Backend::ORT,
                          Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }
  valid_rknpu_backends = {Backend::RKNPU2};
  valid_timvx_backends = {Backend::LITE};
  valid_kunlunxin_backends = {Backend::LITE};
  valid_ascend_backends = {Backend::LITE};
  valid_directml_backends = {Backend::ORT};

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

std::unique_ptr<PaddleSegModel> PaddleSegModel::Clone() const {
  std::unique_ptr<PaddleSegModel> clone_model =
      ultra_infer::utils::make_unique<PaddleSegModel>(PaddleSegModel(*this));
  clone_model->SetRuntime(clone_model->CloneRuntime());
  return clone_model;
}

bool PaddleSegModel::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

bool PaddleSegModel::Predict(cv::Mat *im, SegmentationResult *result) {
  return Predict(*im, result);
}

bool PaddleSegModel::Predict(const cv::Mat &im, SegmentationResult *result) {
  std::vector<SegmentationResult> results;
  if (!BatchPredict({im}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool PaddleSegModel::BatchPredict(const std::vector<cv::Mat> &imgs,
                                  std::vector<SegmentationResult> *results) {
  std::vector<FDMat> fd_images = WrapMat(imgs);
  // Record the shape of input images
  std::map<std::string, std::vector<std::array<int, 2>>> imgs_info;
  preprocessor_.SetImgsInfo(&imgs_info);
  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }
  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  if (!postprocessor_.Run(reused_output_tensors_, results, imgs_info)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}
} // namespace segmentation
} // namespace vision
} // namespace ultra_infer
