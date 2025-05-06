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

#include "ultra_infer/vision/facedet/contrib/ultraface.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace facedet {

UltraFace::UltraFace(const std::string &model_file,
                     const std::string &params_file,
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

bool UltraFace::Initialize() {
  // parameters for preprocess
  size = {320, 240};

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  // Check if the input shape is dynamic after Runtime already initialized,
  is_dynamic_input_ = false;
  auto shape = InputInfoOfRuntime(0).shape;
  for (int i = 0; i < shape.size(); ++i) {
    // if height or width is dynamic
    if (i >= 2 && shape[i] <= 0) {
      is_dynamic_input_ = true;
      break;
    }
  }
  return true;
}

bool UltraFace::Preprocess(
    Mat *mat, FDTensor *output,
    std::map<std::string, std::array<float, 2>> *im_info) {
  // ultraface's preprocess steps
  // 1. resize
  // 2. BGR->RGB
  // 3. HWC->CHW
  int resize_w = size[0];
  int resize_h = size[1];
  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Resize::Run(mat, resize_w, resize_h);
  }

  BGR2RGB::Run(mat);
  // Compute `result = mat * alpha + beta` directly by channel
  // Reference: detect_imgs_onnx.py#L73
  std::vector<float> alpha = {1.0f / 128.0f, 1.0f / 128.0f, 1.0f / 128.0f};
  std::vector<float> beta = {-127.0f * (1.0f / 128.0f),
                             -127.0f * (1.0f / 128.0f),
                             -127.0f * (1.0f / 128.0f)}; // RGB;
  Convert::Run(mat, alpha, beta);

  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {static_cast<float>(mat->Height()),
                                static_cast<float>(mat->Width())};

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");
  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1); // reshape to n, c, h, w
  return true;
}

bool UltraFace::Postprocess(
    std::vector<FDTensor> &infer_result, FaceDetectionResult *result,
    const std::map<std::string, std::array<float, 2>> &im_info,
    float conf_threshold, float nms_iou_threshold) {
  // ultraface has 2 output tensors, scores & boxes
  FDASSERT(
      (infer_result.size() == 2),
      "The default number of output tensor must be 2 according to ultraface.");
  FDTensor &scores_tensor = infer_result.at(0); // (1,4420,2)
  FDTensor &boxes_tensor = infer_result.at(1);  // (1,4420,4)
  FDASSERT((scores_tensor.shape[0] == 1), "Only support batch =1 now.");
  FDASSERT((boxes_tensor.shape[0] == 1), "Only support batch =1 now.");
  if (scores_tensor.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }
  if (boxes_tensor.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }

  result->Clear();
  // must be setup landmarks_per_face before reserve.
  // ultraface detector does not detect landmarks by default.
  result->landmarks_per_face = 0;
  result->Reserve(boxes_tensor.shape[1]);

  float *scores_ptr = static_cast<float *>(scores_tensor.Data());
  float *boxes_ptr = static_cast<float *>(boxes_tensor.Data());
  const size_t num_bboxes = boxes_tensor.shape[1]; // e.g 4420
  // fetch original image shape
  auto iter_ipt = im_info.find("input_shape");
  FDASSERT((iter_ipt != im_info.end()),
           "Cannot find input_shape from im_info.");
  float ipt_h = iter_ipt->second[0];
  float ipt_w = iter_ipt->second[1];

  // decode bounding boxes
  for (size_t i = 0; i < num_bboxes; ++i) {
    float confidence = scores_ptr[2 * i + 1];
    // filter boxes by conf_threshold
    if (confidence <= conf_threshold) {
      continue;
    }
    float x1 = boxes_ptr[4 * i + 0] * ipt_w;
    float y1 = boxes_ptr[4 * i + 1] * ipt_h;
    float x2 = boxes_ptr[4 * i + 2] * ipt_w;
    float y2 = boxes_ptr[4 * i + 3] * ipt_h;
    result->boxes.emplace_back(std::array<float, 4>{x1, y1, x2, y2});
    result->scores.push_back(confidence);
  }

  if (result->boxes.size() == 0) {
    return true;
  }

  utils::NMS(result, nms_iou_threshold);

  // scale and clip box
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    result->boxes[i][0] = std::max(result->boxes[i][0], 0.0f);
    result->boxes[i][1] = std::max(result->boxes[i][1], 0.0f);
    result->boxes[i][2] = std::max(result->boxes[i][2], 0.0f);
    result->boxes[i][3] = std::max(result->boxes[i][3], 0.0f);
    result->boxes[i][0] = std::min(result->boxes[i][0], ipt_w - 1.0f);
    result->boxes[i][1] = std::min(result->boxes[i][1], ipt_h - 1.0f);
    result->boxes[i][2] = std::min(result->boxes[i][2], ipt_w - 1.0f);
    result->boxes[i][3] = std::min(result->boxes[i][3], ipt_h - 1.0f);
  }
  return true;
}

bool UltraFace::Predict(cv::Mat *im, FaceDetectionResult *result,
                        float conf_threshold, float nms_iou_threshold) {
  Mat mat(*im);
  std::vector<FDTensor> input_tensors(1);

  std::map<std::string, std::array<float, 2>> im_info;

  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {static_cast<float>(mat.Height()),
                            static_cast<float>(mat.Width())};
  im_info["output_shape"] = {static_cast<float>(mat.Height()),
                             static_cast<float>(mat.Width())};

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

  if (!Postprocess(output_tensors, result, im_info, conf_threshold,
                   nms_iou_threshold)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }
  return true;
}

} // namespace facedet
} // namespace vision
} // namespace ultra_infer
