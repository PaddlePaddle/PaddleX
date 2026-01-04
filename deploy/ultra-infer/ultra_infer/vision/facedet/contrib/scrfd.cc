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

#include "ultra_infer/vision/facedet/contrib/scrfd.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace facedet {

void SCRFD::LetterBox(Mat *mat, const std::vector<int> &size,
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

SCRFD::SCRFD(const std::string &model_file, const std::string &params_file,
             const RuntimeOption &custom_option,
             const ModelFormat &model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
    valid_rknpu_backends = {Backend::RKNPU2};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool SCRFD::Initialize() {
  // parameters for preprocess
  use_kps = true;
  size = {640, 640};
  padding_value = {0.0, 0.0, 0.0};
  is_mini_pad = false;
  is_no_pad = false;
  is_scale_up = false;
  stride = 32;
  downsample_strides = {8, 16, 32};
  num_anchors = 2;
  landmarks_per_face = 5;
  center_points_is_update_ = false;
  max_nms = 30000;
  // num_outputs = use_kps ? 9 : 6;
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

bool SCRFD::Preprocess(Mat *mat, FDTensor *output,
                       std::map<std::string, std::array<float, 2>> *im_info) {
  float ratio = std::min(size[1] * 1.0f / static_cast<float>(mat->Height()),
                         size[0] * 1.0f / static_cast<float>(mat->Width()));
  if (std::fabs(ratio - 1.0f) > 1e-06) {
    int interp = cv::INTER_LINEAR;
    if (ratio > 1.0) {
      interp = cv::INTER_LINEAR;
    }
    int resize_h = int(mat->Height() * ratio);
    int resize_w = int(mat->Width() * ratio);
    Resize::Run(mat, resize_w, resize_h, -1, -1, interp);
  }
  // scrfd's preprocess steps
  // 1. letterbox
  // 2. BGR->RGB
  // 3. HWC->CHW
  SCRFD::LetterBox(mat, size, padding_value, is_mini_pad, is_no_pad,
                   is_scale_up, stride);

  BGR2RGB::Run(mat);
  if (!disable_normalize_) {
    // Normalize::Run(mat, std::vector<float>(mat->Channels(), 0.0),
    //                std::vector<float>(mat->Channels(), 1.0));
    // Compute `result = mat * alpha + beta` directly by channel
    // Original Repo/tools/scrfd.py: cv2.dnn.blobFromImage(img, 1.0/128,
    // input_size, (127.5, 127.5, 127.5), swapRB=True)
    std::vector<float> alpha = {1.f / 128.f, 1.f / 128.f, 1.f / 128.f};
    std::vector<float> beta = {-127.5f / 128.f, -127.5f / 128.f,
                               -127.5f / 128.f};
    Convert::Run(mat, alpha, beta);
  }

  if (!disable_permute_) {
    HWC2CHW::Run(mat);
    Cast::Run(mat, "float");
  }

  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {static_cast<float>(mat->Height()),
                                static_cast<float>(mat->Width())};
  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1); // reshape to n, c, h, w
  return true;
}

void SCRFD::GeneratePoints() {
  if (center_points_is_update_ && !is_dynamic_input_) {
    return;
  }
  // 8, 16, 32
  for (auto local_stride : downsample_strides) {
    unsigned int num_grid_w = size[0] / local_stride;
    unsigned int num_grid_h = size[1] / local_stride;
    // y
    for (unsigned int i = 0; i < num_grid_h; ++i) {
      // x
      for (unsigned int j = 0; j < num_grid_w; ++j) {
        // num_anchors, col major
        for (unsigned int k = 0; k < num_anchors; ++k) {
          SCRFDPoint point;
          point.cx = static_cast<float>(j);
          point.cy = static_cast<float>(i);
          center_points_[local_stride].push_back(point);
        }
      }
    }
  }

  center_points_is_update_ = true;
}

bool SCRFD::Postprocess(
    std::vector<FDTensor> &infer_result, FaceDetectionResult *result,
    const std::map<std::string, std::array<float, 2>> &im_info,
    float conf_threshold, float nms_iou_threshold) {
  // number of downsample_strides
  int fmc = downsample_strides.size();
  // scrfd has 6,9,10,15 output tensors
  FDASSERT((infer_result.size() == 9 || infer_result.size() == 6 ||
            infer_result.size() == 10 || infer_result.size() == 15),
           "The default number of output tensor must be 6, 9, 10, or 15 "
           "according to scrfd.");
  FDASSERT((fmc == 3 || fmc == 5), "The fmc must be 3 or 5");
  FDASSERT((infer_result.at(0).shape[0] == 1), "Only support batch =1 now.");
  for (int i = 0; i < fmc; ++i) {
    if (infer_result.at(i).dtype != FDDataType::FP32) {
      FDERROR << "Only support post process with float32 data." << std::endl;
      return false;
    }
  }
  int total_num_boxes = 0;
  // compute the reserve space.
  for (int f = 0; f < fmc; ++f) {
    total_num_boxes += infer_result.at(f).shape[1];
  };
  GeneratePoints();
  result->Clear();
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
  if (!is_scale_up) {
    scale = std::min(scale, 1.0f);
  }
  float pad_h = (out_h - ipt_h * scale) / 2.0f;
  float pad_w = (out_w - ipt_w * scale) / 2.0f;
  if (is_mini_pad) {
    pad_h = static_cast<float>(static_cast<int>(pad_h) % stride);
    pad_w = static_cast<float>(static_cast<int>(pad_w) % stride);
  }
  // must be setup landmarks_per_face before reserve
  if (use_kps) {
    result->landmarks_per_face = landmarks_per_face;
  } else {
    // force landmarks_per_face = 0, if use_kps has been set as 'false'.
    result->landmarks_per_face = 0;
  }

  result->Reserve(total_num_boxes);
  unsigned int count = 0;
  // loop each stride
  for (int f = 0; f < fmc; ++f) {
    float *score_ptr = static_cast<float *>(infer_result.at(f).Data());
    float *bbox_ptr = static_cast<float *>(infer_result.at(f + fmc).Data());
    const unsigned int num_points = infer_result.at(f).shape[1];
    int current_stride = downsample_strides[f];
    auto &stride_points = center_points_[current_stride];
    // loop each anchor
    for (unsigned int i = 0; i < num_points; ++i) {
      const float cls_conf = score_ptr[i];
      if (cls_conf < conf_threshold)
        continue; // filter
      auto &point = stride_points.at(i);
      const float cx = point.cx; // cx
      const float cy = point.cy; // cy
      // bbox
      const float *offsets = bbox_ptr + i * 4;
      float l = offsets[0]; // left
      float t = offsets[1]; // top
      float r = offsets[2]; // right
      float b = offsets[3]; // bottom

      float x1 = ((cx - l) * static_cast<float>(current_stride) -
                  static_cast<float>(pad_w)) /
                 scale; // cx - l x1
      float y1 = ((cy - t) * static_cast<float>(current_stride) -
                  static_cast<float>(pad_h)) /
                 scale; // cy - t y1
      float x2 = ((cx + r) * static_cast<float>(current_stride) -
                  static_cast<float>(pad_w)) /
                 scale; // cx + r x2
      float y2 = ((cy + b) * static_cast<float>(current_stride) -
                  static_cast<float>(pad_h)) /
                 scale; // cy + b y2
      result->boxes.emplace_back(std::array<float, 4>{x1, y1, x2, y2});
      result->scores.push_back(cls_conf);
      if (use_kps) {
        float *landmarks_ptr =
            static_cast<float *>(infer_result.at(f + 2 * fmc).Data());
        // landmarks
        const float *kps_offsets = landmarks_ptr + i * (landmarks_per_face * 2);
        for (unsigned int j = 0; j < landmarks_per_face * 2; j += 2) {
          float kps_l = kps_offsets[j];
          float kps_t = kps_offsets[j + 1];
          float kps_x = ((cx + kps_l) * static_cast<float>(current_stride) -
                         static_cast<float>(pad_w)) /
                        scale; // cx + l x
          float kps_y = ((cy + kps_t) * static_cast<float>(current_stride) -
                         static_cast<float>(pad_h)) /
                        scale; // cy + t y
          result->landmarks.emplace_back(std::array<float, 2>{kps_x, kps_y});
        }
      }
      count += 1; // limit boxes for nms.
      if (count > max_nms) {
        break;
      }
    }
  }

  // fetch original image shape
  FDASSERT((iter_ipt != im_info.end()),
           "Cannot find input_shape from im_info.");

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
  // scale and clip landmarks
  if (use_kps) {
    for (size_t i = 0; i < result->landmarks.size(); ++i) {
      result->landmarks[i][0] = std::max(result->landmarks[i][0], 0.0f);
      result->landmarks[i][1] = std::max(result->landmarks[i][1], 0.0f);
      result->landmarks[i][0] = std::min(result->landmarks[i][0], ipt_w - 1.0f);
      result->landmarks[i][1] = std::min(result->landmarks[i][1], ipt_h - 1.0f);
    }
  }
  return true;
}

bool SCRFD::Predict(cv::Mat *im, FaceDetectionResult *result,
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

void SCRFD::DisableNormalize() { disable_normalize_ = true; }

void SCRFD::DisablePermute() { disable_permute_ = true; }
} // namespace facedet
} // namespace vision
} // namespace ultra_infer
