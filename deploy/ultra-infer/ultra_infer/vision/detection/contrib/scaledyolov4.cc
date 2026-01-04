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

#include "ultra_infer/vision/detection/contrib/scaledyolov4.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace detection {

void ScaledYOLOv4::LetterBox(Mat *mat, const std::vector<int> &size,
                             const std::vector<float> &color, bool _auto,
                             bool scale_fill, bool scale_up, int stride) {
  float scale =
      std::min(size[1] * 1.0 / mat->Height(), size[0] * 1.0 / mat->Width());
  if (!scale_up) {
    scale = std::min(scale, 1.0f);
  }

  int resize_h = int(round(mat->Height() * scale));
  int resize_w = int(round(mat->Width() * scale));

  int pad_w = size[0] - resize_w;
  int pad_h = size[1] - resize_h;
  if (_auto) {
    pad_h = pad_h % stride;
    pad_w = pad_w % stride;
  } else if (scale_fill) {
    pad_h = 0;
    pad_w = 0;
    resize_h = size[1];
    resize_w = size[0];
  }
  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Resize::Run(mat, resize_w, resize_h);
  }
  if (pad_h > 0 || pad_w > 0) {
    float half_h = pad_h * 1.0 / 2;
    int top = int(round(half_h - 0.1));
    int bottom = int(round(half_h + 0.1));
    float half_w = pad_w * 1.0 / 2;
    int left = int(round(half_w - 0.1));
    int right = int(round(half_w + 0.1));
    Pad::Run(mat, top, bottom, left, right, color);
  }
}

ScaledYOLOv4::ScaledYOLOv4(const std::string &model_file,
                           const std::string &params_file,
                           const RuntimeOption &custom_option,
                           const ModelFormat &model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  } else {
    valid_cpu_backends = {Backend::PDINFER};
    valid_gpu_backends = {Backend::PDINFER};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool ScaledYOLOv4::Initialize() {
  // parameters for preprocess
  size = {640, 640};
  padding_value = {114.0, 114.0, 114.0};
  is_mini_pad = false;
  is_no_pad = false;
  is_scale_up = false;
  stride = 32;
  max_wh = 7680.0;
  reused_input_tensors_.resize(1);

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  // Check if the input shape is dynamic after Runtime already initialized,
  // Note that, We need to force is_mini_pad 'false' to keep static
  // shape after padding (LetterBox) when the is_dynamic_shape is 'false'.
  is_dynamic_input_ = false;
  auto shape = InputInfoOfRuntime(0).shape;
  for (int i = 0; i < shape.size(); ++i) {
    // if height or width is dynamic
    if (i >= 2 && shape[i] <= 0) {
      is_dynamic_input_ = true;
      break;
    }
  }
  if (!is_dynamic_input_) {
    is_mini_pad = false;
  }
  return true;
}

bool ScaledYOLOv4::Preprocess(
    Mat *mat, FDTensor *output,
    std::map<std::string, std::array<float, 2>> *im_info) {
  // process after image load
  float ratio = std::min(size[1] * 1.0f / static_cast<float>(mat->Height()),
                         size[0] * 1.0f / static_cast<float>(mat->Width()));
  if (std::fabs(ratio - 1.0f) > 1e-06) {
    int interp = cv::INTER_AREA;
    if (ratio > 1.0) {
      interp = cv::INTER_LINEAR;
    }
    int resize_h = int(mat->Height() * ratio);
    int resize_w = int(mat->Width() * ratio);
    Resize::Run(mat, resize_w, resize_h, -1, -1, interp);
  }
  // ScaledYOLOv4's preprocess steps
  // 1. letterbox
  // 2. BGR->RGB
  // 3. HWC->CHW
  ScaledYOLOv4::LetterBox(mat, size, padding_value, is_mini_pad, is_no_pad,
                          is_scale_up, stride);
  BGR2RGB::Run(mat);
  // Normalize::Run(mat, std::vector<float>(mat->Channels(), 0.0),
  //                std::vector<float>(mat->Channels(), 1.0));
  // Compute `result = mat * alpha + beta` directly by channel
  std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  std::vector<float> beta = {0.0f, 0.0f, 0.0f};
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

bool ScaledYOLOv4::Postprocess(
    FDTensor &infer_result, DetectionResult *result,
    const std::map<std::string, std::array<float, 2>> &im_info,
    float conf_threshold, float nms_iou_threshold) {
  FDASSERT(infer_result.shape[0] == 1, "Only support batch =1 now.");
  result->Clear();
  result->Reserve(infer_result.shape[1]);
  if (infer_result.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }
  float *data = static_cast<float *>(infer_result.Data());
  for (size_t i = 0; i < infer_result.shape[1]; ++i) {
    int s = i * infer_result.shape[2];
    float confidence = data[s + 4];
    float *max_class_score =
        std::max_element(data + s + 5, data + s + infer_result.shape[2]);
    confidence *= (*max_class_score);
    // filter boxes by conf_threshold
    if (confidence <= conf_threshold) {
      continue;
    }
    int32_t label_id = std::distance(data + s + 5, max_class_score);
    // convert from [x, y, w, h] to [x1, y1, x2, y2]
    result->boxes.emplace_back(std::array<float, 4>{
        data[s] - data[s + 2] / 2.0f + label_id * max_wh,
        data[s + 1] - data[s + 3] / 2.0f + label_id * max_wh,
        data[s + 0] + data[s + 2] / 2.0f + label_id * max_wh,
        data[s + 1] + data[s + 3] / 2.0f + label_id * max_wh});
    result->label_ids.push_back(label_id);
    result->scores.push_back(confidence);
  }
  utils::NMS(result, nms_iou_threshold);

  // scale the boxes to the origin image shape
  auto iter_out = im_info.find("output_shape");
  auto iter_ipt = im_info.find("input_shape");
  FDASSERT(iter_out != im_info.end() && iter_ipt != im_info.end(),
           "Cannot find input_shape or output_shape from im_info.");
  float out_h = iter_out->second[0];
  float out_w = iter_out->second[1];
  float ipt_h = iter_ipt->second[0];
  float ipt_w = iter_ipt->second[1];
  float scale = std::min(out_h / ipt_h, out_w / ipt_w);
  float pad_h = (out_h - ipt_h * scale) / 2.0f;
  float pad_w = (out_w - ipt_w * scale) / 2.0f;
  if (is_mini_pad) {
    // 和 LetterBox中_auto=true的处理逻辑对应
    pad_h = static_cast<float>(static_cast<int>(pad_h) % stride);
    pad_w = static_cast<float>(static_cast<int>(pad_w) % stride);
  }
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    int32_t label_id = (result->label_ids)[i];
    // clip box
    result->boxes[i][0] = result->boxes[i][0] - max_wh * label_id;
    result->boxes[i][1] = result->boxes[i][1] - max_wh * label_id;
    result->boxes[i][2] = result->boxes[i][2] - max_wh * label_id;
    result->boxes[i][3] = result->boxes[i][3] - max_wh * label_id;
    result->boxes[i][0] = std::max((result->boxes[i][0] - pad_w) / scale, 0.0f);
    result->boxes[i][1] = std::max((result->boxes[i][1] - pad_h) / scale, 0.0f);
    result->boxes[i][2] = std::max((result->boxes[i][2] - pad_w) / scale, 0.0f);
    result->boxes[i][3] = std::max((result->boxes[i][3] - pad_h) / scale, 0.0f);
    result->boxes[i][0] = std::min(result->boxes[i][0], ipt_w - 1.0f);
    result->boxes[i][1] = std::min(result->boxes[i][1], ipt_h - 1.0f);
    result->boxes[i][2] = std::min(result->boxes[i][2], ipt_w - 1.0f);
    result->boxes[i][3] = std::min(result->boxes[i][3], ipt_h - 1.0f);
  }
  return true;
}

bool ScaledYOLOv4::Predict(cv::Mat *im, DetectionResult *result,
                           float conf_threshold, float nms_iou_threshold) {
  Mat mat(*im);

  std::map<std::string, std::array<float, 2>> im_info;

  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {static_cast<float>(mat.Height()),
                            static_cast<float>(mat.Width())};
  im_info["output_shape"] = {static_cast<float>(mat.Height()),
                             static_cast<float>(mat.Width())};

  if (!Preprocess(&mat, &reused_input_tensors_[0], &im_info)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  if (!Infer()) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }
  if (!Postprocess(reused_output_tensors_[0], result, im_info, conf_threshold,
                   nms_iou_threshold)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }

  return true;
}

} // namespace detection
} // namespace vision
} // namespace ultra_infer
