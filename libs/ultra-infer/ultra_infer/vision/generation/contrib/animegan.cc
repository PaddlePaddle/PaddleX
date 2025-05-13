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

#include "ultra_infer/vision/generation/contrib/animegan.h"
#include "ultra_infer/function/functions.h"

namespace ultra_infer {
namespace vision {
namespace generation {

AnimeGAN::AnimeGAN(const std::string &model_file,
                   const std::string &params_file,
                   const RuntimeOption &custom_option,
                   const ModelFormat &model_format) {

  valid_cpu_backends = {Backend::PDINFER};
  valid_gpu_backends = {Backend::PDINFER};

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;

  initialized = Initialize();
}

bool AnimeGAN::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

bool AnimeGAN::Predict(cv::Mat &img, cv::Mat *result) {
  std::vector<cv::Mat> results;
  if (!BatchPredict({img}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool AnimeGAN::BatchPredict(const std::vector<cv::Mat> &images,
                            std::vector<cv::Mat> *results) {
  std::vector<FDMat> fd_images = WrapMat(images);
  std::vector<FDTensor> processed_data(1);
  if (!preprocessor_.Run(fd_images, &(processed_data))) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }
  std::vector<FDTensor> infer_result(1);
  processed_data[0].name = InputInfoOfRuntime(0).name;

  if (!Infer(processed_data, &infer_result)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }
  if (!postprocessor_.Run(infer_result, results)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

} // namespace generation
} // namespace vision
} // namespace ultra_infer
