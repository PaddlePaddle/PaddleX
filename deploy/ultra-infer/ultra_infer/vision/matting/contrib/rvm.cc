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

#include "ultra_infer/vision/matting/contrib/rvm.h"

#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace matting {

RobustVideoMatting::RobustVideoMatting(const std::string &model_file,
                                       const std::string &params_file,
                                       const RuntimeOption &custom_option,
                                       const ModelFormat &model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT, Backend::OPENVINO};
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

bool RobustVideoMatting::Initialize() {
  // parameters for preprocess
  size = {1080, 1920};

  video_mode = true;

  swap_rb = true;

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

bool RobustVideoMatting::Preprocess(
    Mat *mat, FDTensor *output,
    std::map<std::string, std::array<int, 2>> *im_info) {
  // Resize
  int resize_w = size[0];
  int resize_h = size[1];
  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Resize::Run(mat, resize_w, resize_h);
  }
  // Convert_and_permute(swap_rb=true)
  std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  std::vector<float> beta = {0.0f, 0.0f, 0.0f};
  ConvertAndPermute::Run(mat, alpha, beta, swap_rb);

  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {mat->Height(), mat->Width()};

  mat->ShareWithTensor(output);
  output->ExpandDim(0); // reshape to n, c, h, w
  return true;
}

bool RobustVideoMatting::Postprocess(
    std::vector<FDTensor> &infer_result, MattingResult *result,
    const std::map<std::string, std::array<int, 2>> &im_info) {
  FDASSERT((infer_result.size() == 6),
           "The default number of output tensor must be 6 according to "
           "RobustVideoMatting.");
  FDTensor &fgr = infer_result.at(0);   // fgr (1, 3, h, w) 0.~1.
  FDTensor &alpha = infer_result.at(1); // alpha (1, 1, h, w) 0.~1.
  FDASSERT((fgr.shape[0] == 1), "Only support batch = 1 now.");
  FDASSERT((alpha.shape[0] == 1), "Only support batch = 1 now.");
  if (fgr.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }
  if (alpha.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }
  // update context
  if (video_mode) {
    for (size_t i = 0; i < 4; ++i) {
      FDTensor &rki = infer_result.at(i + 2);
      dynamic_inputs_dims_[i] = rki.shape;
      dynamic_inputs_datas_[i].resize(rki.Numel());
      memcpy(dynamic_inputs_datas_[i].data(), rki.Data(),
             rki.Numel() * FDDataTypeSize(rki.dtype));
    }
  }

  auto iter_in = im_info.find("input_shape");
  auto iter_out = im_info.find("output_shape");
  FDASSERT(iter_out != im_info.end() && iter_in != im_info.end(),
           "Cannot find input_shape or output_shape from im_info.");
  int out_h = iter_out->second[0];
  int out_w = iter_out->second[1];
  int in_h = iter_in->second[0];
  int in_w = iter_in->second[1];

  // for alpha
  float *alpha_ptr = static_cast<float *>(alpha.Data());
  Mat alpha_resized = Mat::Create(out_h, out_w, 1, FDDataType::FP32,
                                  alpha_ptr); // ref-only, zero copy.
  if ((out_h != in_h) || (out_w != in_w)) {
    Resize::Run(&alpha_resized, in_w, in_h, -1, -1);
  }

  // for foreground
  float *fgr_ptr = static_cast<float *>(fgr.Data());
  Mat fgr_resized = Mat::Create(out_h, out_w, 1, FDDataType::FP32,
                                fgr_ptr); // ref-only, zero copy.
  if ((out_h != in_h) || (out_w != in_w)) {
    Resize::Run(&fgr_resized, in_w, in_h, -1, -1);
  }

  result->contain_foreground = true;
  // if contain_foreground == true, shape must set to (h, w, c)
  result->shape = {static_cast<int64_t>(in_h), static_cast<int64_t>(in_w), 3};
  int numel = in_h * in_w;
  int nbytes = numel * sizeof(float);
  result->Resize(numel);
  memcpy(result->alpha.data(), alpha_resized.Data(), nbytes);
  memcpy(result->foreground.data(), fgr_resized.Data(), nbytes);
  return true;
}

bool RobustVideoMatting::Predict(cv::Mat *im, MattingResult *result) {
  Mat mat(*im);
  int inputs_nums = NumInputsOfRuntime();
  std::vector<FDTensor> input_tensors(inputs_nums);
  std::map<std::string, std::array<int, 2>> im_info;
  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {mat.Height(), mat.Width()};
  im_info["output_shape"] = {mat.Height(), mat.Width()};
  // convert vector to FDTensor
  for (size_t i = 1; i < inputs_nums; ++i) {
    input_tensors[i].SetExternalData(dynamic_inputs_dims_[i - 1],
                                     FDDataType::FP32,
                                     dynamic_inputs_datas_[i - 1].data());
    input_tensors[i].device = Device::CPU;
  }
  if (!Preprocess(&mat, &input_tensors[0], &im_info)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }
  for (size_t i = 0; i < inputs_nums; ++i) {
    input_tensors[i].name = InputInfoOfRuntime(i).name;
  }
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
