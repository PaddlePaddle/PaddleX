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

#include "ultra_infer/vision/ocr/ppocr/structurev2_ser_vi_layoutxlm.h"

#include "ultra_infer/utils/unique_ptr.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

StructureV2SERViLayoutXLMModel::StructureV2SERViLayoutXLMModel(
    const std::string &model_file, const std::string &params_file,
    const std::string &config_file, const RuntimeOption &custom_option,
    const ModelFormat &model_format) {
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
    valid_horizon_backends = {Backend::HORIZONNPU};
  }

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

std::unique_ptr<StructureV2SERViLayoutXLMModel>
StructureV2SERViLayoutXLMModel::Clone() const {
  std::unique_ptr<StructureV2SERViLayoutXLMModel> clone_model =
      utils::make_unique<StructureV2SERViLayoutXLMModel>(
          StructureV2SERViLayoutXLMModel(*this));
  clone_model->SetRuntime(clone_model->CloneRuntime());
  return clone_model;
}

bool StructureV2SERViLayoutXLMModel::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
