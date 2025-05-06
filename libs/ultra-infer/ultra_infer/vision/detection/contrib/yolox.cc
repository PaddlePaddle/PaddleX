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

#include "ultra_infer/vision/detection/contrib/yolox.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace detection {

struct YOLOXAnchor {
  int grid0;
  int grid1;
  int stride;
};

void GenerateYOLOXAnchors(const std::vector<int> &size,
                          const std::vector<int> &downsample_strides,
                          std::vector<YOLOXAnchor> *anchors) {
  // size: tuple of input (width, height)
  // downsample_strides: downsample strides in YOLOX, e.g (8,16,32)
  const int width = size[0];
  const int height = size[1];
  for (const auto &ds : downsample_strides) {
    int num_grid_w = width / ds;
    int num_grid_h = height / ds;
    for (int g1 = 0; g1 < num_grid_h; ++g1) {
      for (int g0 = 0; g0 < num_grid_w; ++g0) {
        (*anchors).emplace_back(YOLOXAnchor{g0, g1, ds});
      }
    }
  }
}

void LetterBoxWithRightBottomPad(Mat *mat, std::vector<int> size,
                                 std::vector<float> color) {
  // specific pre process for YOLOX, not the same as YOLOv5
  // reference: YOLOX/yolox/data/data_augment.py#L142
  float r = std::min(size[1] * 1.0f / static_cast<float>(mat->Height()),
                     size[0] * 1.0f / static_cast<float>(mat->Width()));

  int resize_h = int(round(static_cast<float>(mat->Height()) * r));
  int resize_w = int(round(static_cast<float>(mat->Width()) * r));

  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Resize::Run(mat, resize_w, resize_h);
  }

  int pad_w = size[0] - resize_w;
  int pad_h = size[1] - resize_h;
  // right-bottom padding for YOLOX
  if (pad_h > 0 || pad_w > 0) {
    int top = 0;
    int left = 0;
    int right = pad_w;
    int bottom = pad_h;
    Pad::Run(mat, top, bottom, left, right, color);
  }
}

YOLOX::YOLOX(const std::string &model_file, const std::string &params_file,
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

bool YOLOX::Initialize() {
  // parameters for preprocess
  size = {640, 640};
  padding_value = {114.0, 114.0, 114.0};
  downsample_strides = {8, 16, 32};
  max_wh = 4096.0f;
  is_decode_exported = false;
  reused_input_tensors_.resize(1);

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  // Check if the input shape is dynamic after Runtime already initialized.
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

bool YOLOX::Preprocess(Mat *mat, FDTensor *output,
                       std::map<std::string, std::array<float, 2>> *im_info) {
  // YOLOX ( >= v0.1.1) preprocess steps
  // 1. preproc
  // 2. HWC->CHW
  // 3. NO!!! BRG2GRB and Normalize needed in YOLOX
  LetterBoxWithRightBottomPad(mat, size, padding_value);
  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {static_cast<float>(mat->Height()),
                                static_cast<float>(mat->Width())};

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");
  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1); // reshape to n, c, h, w
  return true;
}

bool YOLOX::Postprocess(
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
  float r = std::min(out_h / ipt_h, out_w / ipt_w);
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    int32_t label_id = (result->label_ids)[i];
    // clip box
    result->boxes[i][0] = result->boxes[i][0] - max_wh * label_id;
    result->boxes[i][1] = result->boxes[i][1] - max_wh * label_id;
    result->boxes[i][2] = result->boxes[i][2] - max_wh * label_id;
    result->boxes[i][3] = result->boxes[i][3] - max_wh * label_id;
    result->boxes[i][0] = std::max(result->boxes[i][0] / r, 0.0f);
    result->boxes[i][1] = std::max(result->boxes[i][1] / r, 0.0f);
    result->boxes[i][2] = std::max(result->boxes[i][2] / r, 0.0f);
    result->boxes[i][3] = std::max(result->boxes[i][3] / r, 0.0f);
    result->boxes[i][0] = std::min(result->boxes[i][0], ipt_w - 1.0f);
    result->boxes[i][1] = std::min(result->boxes[i][1], ipt_h - 1.0f);
    result->boxes[i][2] = std::min(result->boxes[i][2], ipt_w - 1.0f);
    result->boxes[i][3] = std::min(result->boxes[i][3], ipt_h - 1.0f);
  }
  return true;
}

bool YOLOX::PostprocessWithDecode(
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
  // generate anchors with dowmsample strides
  std::vector<YOLOXAnchor> anchors;
  GenerateYOLOXAnchors(size, downsample_strides, &anchors);

  // infer_result shape might look like (1,n,85=5+80)
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
    // fetch i-th anchor
    float grid0 = static_cast<float>(anchors.at(i).grid0);
    float grid1 = static_cast<float>(anchors.at(i).grid1);
    float downsample_stride = static_cast<float>(anchors.at(i).stride);
    // convert from offsets to [x, y, w, h]
    float dx = data[s];
    float dy = data[s + 1];
    float dw = data[s + 2];
    float dh = data[s + 3];

    float x = (dx + grid0) * downsample_stride;
    float y = (dy + grid1) * downsample_stride;
    float w = std::exp(dw) * downsample_stride;
    float h = std::exp(dh) * downsample_stride;

    // convert from [x, y, w, h] to [x1, y1, x2, y2]
    result->boxes.emplace_back(std::array<float, 4>{
        x - w / 2.0f + label_id * max_wh, y - h / 2.0f + label_id * max_wh,
        x + w / 2.0f + label_id * max_wh, y + h / 2.0f + label_id * max_wh});
    // label_id * max_wh for multi classes NMS
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
  float r = std::min(out_h / ipt_h, out_w / ipt_w);
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    int32_t label_id = (result->label_ids)[i];
    // clip box
    result->boxes[i][0] = result->boxes[i][0] - max_wh * label_id;
    result->boxes[i][1] = result->boxes[i][1] - max_wh * label_id;
    result->boxes[i][2] = result->boxes[i][2] - max_wh * label_id;
    result->boxes[i][3] = result->boxes[i][3] - max_wh * label_id;
    result->boxes[i][0] = std::max(result->boxes[i][0] / r, 0.0f);
    result->boxes[i][1] = std::max(result->boxes[i][1] / r, 0.0f);
    result->boxes[i][2] = std::max(result->boxes[i][2] / r, 0.0f);
    result->boxes[i][3] = std::max(result->boxes[i][3] / r, 0.0f);
    result->boxes[i][0] = std::min(result->boxes[i][0], ipt_w - 1.0f);
    result->boxes[i][1] = std::min(result->boxes[i][1], ipt_h - 1.0f);
    result->boxes[i][2] = std::min(result->boxes[i][2], ipt_w - 1.0f);
    result->boxes[i][3] = std::min(result->boxes[i][3], ipt_h - 1.0f);
  }
  return true;
}

bool YOLOX::Predict(cv::Mat *im, DetectionResult *result, float conf_threshold,
                    float nms_iou_threshold) {
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

  if (is_decode_exported) {
    if (!Postprocess(reused_output_tensors_[0], result, im_info, conf_threshold,
                     nms_iou_threshold)) {
      FDERROR << "Failed to post process." << std::endl;
      return false;
    }
  } else {
    if (!PostprocessWithDecode(reused_output_tensors_[0], result, im_info,
                               conf_threshold, nms_iou_threshold)) {
      FDERROR << "Failed to post process." << std::endl;
      return false;
    }
  }
  return true;
}

} // namespace detection
} // namespace vision
} // namespace ultra_infer
