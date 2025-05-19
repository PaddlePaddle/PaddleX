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

#include "ultra_infer/vision/classification/ppshitu/ppshituv2_rec.h"

#include "ultra_infer/utils/unique_ptr.h"

namespace ultra_infer {
namespace vision {
namespace classification {

PPShiTuV2Recognizer::PPShiTuV2Recognizer(const std::string &model_file,
                                         const std::string &params_file,
                                         const std::string &config_file,
                                         const RuntimeOption &custom_option,
                                         const ModelFormat &model_format)
    : preprocessor_(config_file) {
  if (model_format == ModelFormat::PADDLE) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::PDINFER, Backend::ORT,
                          Backend::LITE};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_timvx_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ipu_backends = {Backend::PDINFER};
    valid_directml_backends = {Backend::ORT};
  } else if (model_format == ModelFormat::SOPHGO) {
    valid_sophgonpu_backends = {Backend::SOPHGOTPU};
  } else {
    valid_cpu_backends = {Backend::ORT, Backend::OPENVINO};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
    valid_rknpu_backends = {Backend::RKNPU2};
    valid_directml_backends = {Backend::ORT};
  }

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

std::unique_ptr<PPShiTuV2Recognizer> PPShiTuV2Recognizer::Clone() const {
  std::unique_ptr<PPShiTuV2Recognizer> clone_model =
      utils::make_unique<PPShiTuV2Recognizer>(PPShiTuV2Recognizer(*this));
  clone_model->SetRuntime(clone_model->CloneRuntime());
  return clone_model;
}

bool PPShiTuV2Recognizer::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

bool PPShiTuV2Recognizer::Predict(cv::Mat *im, ClassifyResult *result) {
  if (!Predict(*im, result)) {
    return false;
  }
  return true;
}

bool PPShiTuV2Recognizer::Predict(const cv::Mat &im, ClassifyResult *result) {
  FDMat mat = WrapMat(im);
  return Predict(mat, result);
}

bool PPShiTuV2Recognizer::BatchPredict(const std::vector<cv::Mat> &images,
                                       std::vector<ClassifyResult> *results) {
  std::vector<FDMat> mats = WrapMat(images);
  return BatchPredict(mats, results);
}

bool PPShiTuV2Recognizer::Predict(const FDMat &mat, ClassifyResult *result) {
  std::vector<ClassifyResult> results;
  std::vector<FDMat> mats = {mat};
  if (!BatchPredict(mats, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool PPShiTuV2Recognizer::BatchPredict(const std::vector<FDMat> &mats,
                                       std::vector<ClassifyResult> *results) {
  std::vector<FDMat> fd_mats = mats;
  if (!preprocessor_.Run(&fd_mats, &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }
  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
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

} // namespace classification
} // namespace vision
} // namespace ultra_infer
