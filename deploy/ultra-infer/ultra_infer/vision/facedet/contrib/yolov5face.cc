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

#include "ultra_infer/vision/facedet/contrib/yolov5face.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace facedet {

void LetterBox(Mat *mat, std::vector<int> size, std::vector<float> color,
               bool _auto, bool scale_fill = false, bool scale_up = true,
               int stride = 32) {
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

YOLOv5Face::YOLOv5Face(const std::string &model_file,
                       const std::string &params_file,
                       const RuntimeOption &custom_option,
                       const ModelFormat &model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool YOLOv5Face::Initialize() {
  // parameters for preprocess
  size = {640, 640};
  padding_value = {114.0, 114.0, 114.0};
  is_mini_pad = false;
  is_no_pad = false;
  is_scale_up = false;
  stride = 32;
  landmarks_per_face = 5;

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  // Check if the input shape is dynamic after Runtime already initialized,
  // Note that, We need to force is_mini_pad 'false' to keep static
  // shape after padding (LetterBox) when the is_dynamic_input_ is 'false'.
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

bool YOLOv5Face::Preprocess(
    Mat *mat, FDTensor *output,
    std::map<std::string, std::array<float, 2>> *im_info) {
  // process after image load
  float ratio = std::min(size[1] * 1.0f / static_cast<float>(mat->Height()),
                         size[0] * 1.0f / static_cast<float>(mat->Width()));
  if (std::fabs(ratio - 1.0f) > 1e-06) {
    int interp = cv::INTER_LINEAR;
    if (ratio > 1.0) {
      interp = cv::INTER_LINEAR;
    }
    int resize_h = int(round(static_cast<float>(mat->Height()) * ratio));
    int resize_w = int(round(static_cast<float>(mat->Width()) * ratio));
    Resize::Run(mat, resize_w, resize_h, -1, -1, interp);
  }
  // yolov5face's preprocess steps
  // 1. letterbox
  // 2. BGR->RGB
  // 3. HWC->CHW
  LetterBox(mat, size, padding_value, is_mini_pad, is_no_pad, is_scale_up,
            stride);
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

bool YOLOv5Face::Postprocess(
    FDTensor &infer_result, FaceDetectionResult *result,
    const std::map<std::string, std::array<float, 2>> &im_info,
    float conf_threshold, float nms_iou_threshold) {
  // infer_result: (1,n,16) 16=4+1+10+1
  FDASSERT(infer_result.shape[0] == 1, "Only support batch =1 now.");
  if (infer_result.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }

  result->Clear();
  // must be setup landmarks_per_face before reserve
  result->landmarks_per_face = landmarks_per_face;
  result->Reserve(infer_result.shape[1]);

  float *data = static_cast<float *>(infer_result.Data());
  for (size_t i = 0; i < infer_result.shape[1]; ++i) {
    float *reg_cls_ptr = data + (i * infer_result.shape[2]);
    float obj_conf = reg_cls_ptr[4];
    float cls_conf = reg_cls_ptr[15];
    float confidence = obj_conf * cls_conf;
    // filter boxes by conf_threshold
    if (confidence <= conf_threshold) {
      continue;
    }
    float x = reg_cls_ptr[0];
    float y = reg_cls_ptr[1];
    float w = reg_cls_ptr[2];
    float h = reg_cls_ptr[3];

    // convert from [x, y, w, h] to [x1, y1, x2, y2]
    result->boxes.emplace_back(std::array<float, 4>{
        (x - w / 2.f), (y - h / 2.f), (x + w / 2.f), (y + h / 2.f)});
    result->scores.push_back(confidence);
    // decode landmarks (default 5 landmarks)
    if (landmarks_per_face > 0) {
      float *landmarks_ptr = reg_cls_ptr + 5;
      for (size_t j = 0; j < landmarks_per_face * 2; j += 2) {
        result->landmarks.emplace_back(
            std::array<float, 2>{landmarks_ptr[j], landmarks_ptr[j + 1]});
      }
    }
  }

  if (result->boxes.size() == 0) {
    return true;
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
  if (!is_scale_up) {
    scale = std::min(scale, 1.0f);
  }
  float pad_h = (out_h - ipt_h * scale) / 2.f;
  float pad_w = (out_w - ipt_w * scale) / 2.f;
  if (is_mini_pad) {
    pad_h = static_cast<float>(static_cast<int>(pad_h) % stride);
    pad_w = static_cast<float>(static_cast<int>(pad_w) % stride);
  }
  // scale and clip box
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    result->boxes[i][0] = std::max((result->boxes[i][0] - pad_w) / scale, 0.0f);
    result->boxes[i][1] = std::max((result->boxes[i][1] - pad_h) / scale, 0.0f);
    result->boxes[i][2] = std::max((result->boxes[i][2] - pad_w) / scale, 0.0f);
    result->boxes[i][3] = std::max((result->boxes[i][3] - pad_h) / scale, 0.0f);
    result->boxes[i][0] = std::min(result->boxes[i][0], ipt_w - 1.0f);
    result->boxes[i][1] = std::min(result->boxes[i][1], ipt_h - 1.0f);
    result->boxes[i][2] = std::min(result->boxes[i][2], ipt_w - 1.0f);
    result->boxes[i][3] = std::min(result->boxes[i][3], ipt_h - 1.0f);
  }
  // scale and clip landmarks
  for (size_t i = 0; i < result->landmarks.size(); ++i) {
    result->landmarks[i][0] =
        std::max((result->landmarks[i][0] - pad_w) / scale, 0.0f);
    result->landmarks[i][1] =
        std::max((result->landmarks[i][1] - pad_h) / scale, 0.0f);
    result->landmarks[i][0] = std::min(result->landmarks[i][0], ipt_w - 1.0f);
    result->landmarks[i][1] = std::min(result->landmarks[i][1], ipt_h - 1.0f);
  }
  return true;
}

bool YOLOv5Face::Predict(cv::Mat *im, FaceDetectionResult *result,
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

  if (!Postprocess(output_tensors[0], result, im_info, conf_threshold,
                   nms_iou_threshold)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }
  return true;
}

} // namespace facedet
} // namespace vision
} // namespace ultra_infer
