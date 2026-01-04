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

#include "ultra_infer/vision/facealign/contrib/pfld.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace facealign {

PFLD::PFLD(const std::string &model_file, const std::string &params_file,
           const RuntimeOption &custom_option,
           const ModelFormat &model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool PFLD::Initialize() {
  // parameters for preprocess
  size = {112, 112};

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

bool PFLD::Preprocess(Mat *mat, FDTensor *output,
                      std::map<std::string, std::array<int, 2>> *im_info) {
  // Resize
  int resize_w = size[0];
  int resize_h = size[1];
  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Resize::Run(mat, resize_w, resize_h);
  }

  // Normalize
  std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  std::vector<float> beta = {0.0f, 0.0f, 0.0f};
  Convert::Run(mat, alpha, beta);

  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {mat->Height(), mat->Width()};

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");
  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1); // reshape to n, c, h, w
  return true;
}

bool PFLD::Postprocess(
    FDTensor &infer_result, FaceAlignmentResult *result,
    const std::map<std::string, std::array<int, 2>> &im_info) {
  FDASSERT(infer_result.shape[0] == 1, "Only support batch = 1 now.");
  if (infer_result.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }

  auto iter_in = im_info.find("input_shape");
  FDASSERT(iter_in != im_info.end(), "Cannot find input_shape from im_info.");
  int in_h = iter_in->second[0];
  int in_w = iter_in->second[1];

  result->Clear();
  float *data = static_cast<float *>(infer_result.Data());
  for (size_t i = 0; i < infer_result.shape[1]; i += 2) {
    float x = data[i];
    float y = data[i + 1];
    x = std::min(std::max(0.f, x), 1.0f);
    y = std::min(std::max(0.f, y), 1.0f);
    // decode landmarks (default 106 landmarks)
    result->landmarks.emplace_back(std::array<float, 2>{x * in_w, y * in_h});
  }

  return true;
}

bool PFLD::Predict(cv::Mat *im, FaceAlignmentResult *result) {
  Mat mat(*im);
  std::vector<FDTensor> input_tensors(1);

  std::map<std::string, std::array<int, 2>> im_info;

  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {mat.Height(), mat.Width()};
  im_info["output_shape"] = {mat.Height(), mat.Width()};

  if (!Preprocess(&mat, &input_tensors[0], &im_info)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }
  input_tensors[0].name = InputInfoOfRuntime(0).name;
  std::vector<FDTensor> output_tensors;
  if (!Infer(input_tensors, &output_tensors)) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }

  if (!Postprocess(output_tensors[1], result, im_info)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }
  return true;
}

} // namespace facealign
} // namespace vision
} // namespace ultra_infer
