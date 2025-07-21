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

#include "ultra_infer/vision/facedet/contrib/retinaface.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace facedet {

struct RetinaAnchor {
  float cx;
  float cy;
  float s_kx;
  float s_ky;
};

void GenerateRetinaAnchors(const std::vector<int> &size,
                           const std::vector<int> &downsample_strides,
                           const std::vector<std::vector<int>> &min_sizes,
                           std::vector<RetinaAnchor> *anchors) {
  // size: tuple of input (width, height)
  // downsample_strides: downsample strides (steps), e.g (8,16,32)
  // min_sizes: width and height for each anchor,
  // e.g {{16, 32}, {64, 128}, {256, 512}}
  int h = size[1];
  int w = size[0];
  std::vector<std::vector<int>> feature_maps;
  for (auto s : downsample_strides) {
    feature_maps.push_back(
        {static_cast<int>(
             std::ceil(static_cast<float>(h) / static_cast<float>(s))),
         static_cast<int>(
             std::ceil(static_cast<float>(w) / static_cast<float>(s)))});
  }

  (*anchors).clear();
  const size_t num_feature_map = feature_maps.size();
  // reference: layers/functions/prior_box.py#L21
  for (size_t k = 0; k < num_feature_map; ++k) {
    auto f_map = feature_maps.at(k);      // e.g [640//8,640//8]
    auto tmp_min_sizes = min_sizes.at(k); // e.g [8,16]
    int f_h = f_map.at(0);
    int f_w = f_map.at(1);
    for (size_t i = 0; i < f_h; ++i) {
      for (size_t j = 0; j < f_w; ++j) {
        for (auto min_size : tmp_min_sizes) {
          float s_kx =
              static_cast<float>(min_size) / static_cast<float>(w); // e.g 16/w
          float s_ky =
              static_cast<float>(min_size) / static_cast<float>(h); // e.g 16/h
          // (x + 0.5) * step / w normalized loc mapping to input width
          // (y + 0.5) * step / h normalized loc mapping to input height
          float s = static_cast<float>(downsample_strides.at(k));
          float cx = (static_cast<float>(j) + 0.5f) * s / static_cast<float>(w);
          float cy = (static_cast<float>(i) + 0.5f) * s / static_cast<float>(h);
          (*anchors).emplace_back(
              RetinaAnchor{cx, cy, s_kx, s_ky}); // without clip
        }
      }
    }
  }
}

RetinaFace::RetinaFace(const std::string &model_file,
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

bool RetinaFace::Initialize() {
  // parameters for preprocess
  size = {640, 640};
  variance = {0.1f, 0.2f};
  downsample_strides = {8, 16, 32};
  min_sizes = {{16, 32}, {64, 128}, {256, 512}};
  landmarks_per_face = 5;

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

bool RetinaFace::Preprocess(
    Mat *mat, FDTensor *output,
    std::map<std::string, std::array<float, 2>> *im_info) {
  // retinaface's preprocess steps
  // 1. Resize
  // 2. Convert(opencv style) or Normalize
  // 3. HWC->CHW
  int resize_w = size[0];
  int resize_h = size[1];
  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Resize::Run(mat, resize_w, resize_h);
  }

  // Compute `result = mat * alpha + beta` directly by channel
  // Reference: detect.py#L94
  std::vector<float> alpha = {1.f, 1.f, 1.f};
  std::vector<float> beta = {-104.f, -117.f, -123.f}; // BGR;
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

bool RetinaFace::Postprocess(
    std::vector<FDTensor> &infer_result, FaceDetectionResult *result,
    const std::map<std::string, std::array<float, 2>> &im_info,
    float conf_threshold, float nms_iou_threshold) {
  // retinaface has 3 output tensors, boxes & conf & landmarks
  FDASSERT(
      (infer_result.size() == 3),
      "The default number of output tensor must be 3 according to retinaface.");
  FDTensor &boxes_tensor = infer_result.at(0);     // (1,n,4)
  FDTensor &conf_tensor = infer_result.at(1);      // (1,n,2)
  FDTensor &landmarks_tensor = infer_result.at(2); // (1,n,10)
  FDASSERT((boxes_tensor.shape[0] == 1), "Only support batch =1 now.");
  if (boxes_tensor.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }

  result->Clear();
  // must be setup landmarks_per_face before reserve
  result->landmarks_per_face = landmarks_per_face;
  result->Reserve(boxes_tensor.shape[1]);

  float *boxes_ptr = static_cast<float *>(boxes_tensor.Data());
  float *conf_ptr = static_cast<float *>(conf_tensor.Data());
  float *landmarks_ptr = static_cast<float *>(landmarks_tensor.Data());
  const size_t num_bboxes = boxes_tensor.shape[1]; // n
  // fetch original image shape
  auto iter_ipt = im_info.find("input_shape");
  FDASSERT((iter_ipt != im_info.end()),
           "Cannot find input_shape from im_info.");
  float ipt_h = iter_ipt->second[0];
  float ipt_w = iter_ipt->second[1];

  // generate anchors with dowmsample strides
  std::vector<RetinaAnchor> anchors;
  GenerateRetinaAnchors(size, downsample_strides, min_sizes, &anchors);

  // decode bounding boxes
  for (size_t i = 0; i < num_bboxes; ++i) {
    float confidence = conf_ptr[2 * i + 1];
    // filter boxes by conf_threshold
    if (confidence <= conf_threshold) {
      continue;
    }
    float prior_cx = anchors.at(i).cx;
    float prior_cy = anchors.at(i).cy;
    float prior_s_kx = anchors.at(i).s_kx;
    float prior_s_ky = anchors.at(i).s_ky;

    // fetch offsets (dx,dy,dw,dh)
    float dx = boxes_ptr[4 * i + 0];
    float dy = boxes_ptr[4 * i + 1];
    float dw = boxes_ptr[4 * i + 2];
    float dh = boxes_ptr[4 * i + 3];
    // reference: Pytorch_Retinaface/utils/box_utils.py
    float x = prior_cx + dx * variance[0] * prior_s_kx;
    float y = prior_cy + dy * variance[0] * prior_s_ky;
    float w = prior_s_kx * std::exp(dw * variance[1]);
    float h = prior_s_ky * std::exp(dh * variance[1]); // (0.~1.)
    // from (x,y,w,h) to (x1,y1,x2,y2)
    float x1 = (x - w / 2.f) * ipt_w;
    float y1 = (y - h / 2.f) * ipt_h;
    float x2 = (x + w / 2.f) * ipt_w;
    float y2 = (y + h / 2.f) * ipt_h;
    result->boxes.emplace_back(std::array<float, 4>{x1, y1, x2, y2});
    result->scores.push_back(confidence);
    // decode landmarks (default 5 landmarks)
    if (landmarks_per_face > 0) {
      // reference: utils/box_utils.py#L241
      for (size_t j = 0; j < landmarks_per_face * 2; j += 2) {
        float ldx = landmarks_ptr[i * (landmarks_per_face * 2) + (j + 0)];
        float ldy = landmarks_ptr[i * (landmarks_per_face * 2) + (j + 1)];
        float lx = (prior_cx + ldx * variance[0] * prior_s_kx) * ipt_w;
        float ly = (prior_cy + ldy * variance[0] * prior_s_ky) * ipt_h;
        result->landmarks.emplace_back(std::array<float, 2>{lx, ly});
      }
    }
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
  // scale and clip landmarks
  for (size_t i = 0; i < result->landmarks.size(); ++i) {
    result->landmarks[i][0] = std::max(result->landmarks[i][0], 0.0f);
    result->landmarks[i][1] = std::max(result->landmarks[i][1], 0.0f);
    result->landmarks[i][0] = std::min(result->landmarks[i][0], ipt_w - 1.0f);
    result->landmarks[i][1] = std::min(result->landmarks[i][1], ipt_h - 1.0f);
  }
  return true;
}

bool RetinaFace::Predict(cv::Mat *im, FaceDetectionResult *result,
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
