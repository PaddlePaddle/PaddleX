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

#include "ultra_infer/vision/matting/contrib/modnet.h"

#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace matting {

MODNet::MODNet(const std::string &model_file, const std::string &params_file,
               const RuntimeOption &custom_option,
               const ModelFormat &model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT};
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

bool MODNet::Initialize() {
  // parameters for preprocess
  size = {256, 256};
  alpha = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
  beta = {-1.f, -1.f, -1.f}; // RGB
  swap_rb = true;

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

bool MODNet::Preprocess(Mat *mat, FDTensor *output,
                        std::map<std::string, std::array<int, 2>> *im_info) {
  // 1. Resize
  // 2. BGR2RGB
  // 3. Convert(opencv style) or Normalize
  // 4. HWC2CHW
  int resize_w = size[0];
  int resize_h = size[1];
  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Resize::Run(mat, resize_w, resize_h);
  }
  if (swap_rb) {
    BGR2RGB::Run(mat);
  }

  Convert::Run(mat, alpha, beta);
  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {mat->Height(), mat->Width()};

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1); // reshape to n, c, h, w
  return true;
}

bool MODNet::Postprocess(
    std::vector<FDTensor> &infer_result, MattingResult *result,
    const std::map<std::string, std::array<int, 2>> &im_info) {
  FDASSERT((infer_result.size() == 1),
           "The default number of output tensor must be 1 according to "
           "modnet.");
  FDTensor &alpha_tensor = infer_result.at(0); // (1, 1, h, w)
  FDASSERT((alpha_tensor.shape[0] == 1), "Only support batch =1 now.");
  if (alpha_tensor.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }

  auto iter_ipt = im_info.find("input_shape");
  auto iter_out = im_info.find("output_shape");
  FDASSERT(iter_out != im_info.end() && iter_ipt != im_info.end(),
           "Cannot find input_shape or output_shape from im_info.");
  int out_h = iter_out->second[0];
  int out_w = iter_out->second[1];
  int ipt_h = iter_ipt->second[0];
  int ipt_w = iter_ipt->second[1];

  float *alpha_ptr = static_cast<float *>(alpha_tensor.Data());
  // cv::Mat alpha_zero_copy_ref(out_h, out_w, CV_32FC1, alpha_ptr);
  // Mat alpha_resized(alpha_zero_copy_ref);  // ref-only, zero copy.
  Mat alpha_resized = Mat::Create(out_h, out_w, 1, FDDataType::FP32,
                                  alpha_ptr); // ref-only, zero copy.
  if ((out_h != ipt_h) || (out_w != ipt_w)) {
    Resize::Run(&alpha_resized, ipt_w, ipt_h, -1, -1);
  }

  result->Clear();
  // note: must be setup shape before Resize
  result->contain_foreground = false;
  result->shape = {static_cast<int64_t>(ipt_h), static_cast<int64_t>(ipt_w)};
  int numel = ipt_h * ipt_w;
  int nbytes = numel * sizeof(float);
  result->Resize(numel);
  std::memcpy(result->alpha.data(), alpha_resized.Data(), nbytes);
  return true;
}

bool MODNet::Predict(cv::Mat *im, MattingResult *result) {
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

  if (!Postprocess(output_tensors, result, im_info)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }
  return true;
}

} // namespace matting
} // namespace vision
} // namespace ultra_infer
