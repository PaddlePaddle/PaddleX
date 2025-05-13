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

#include "ultra_infer/vision/detection/contrib/nanodet_plus.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace detection {

struct NanoDetPlusCenterPoint {
  int grid0;
  int grid1;
  int stride;
};

void GenerateNanoDetPlusCenterPoints(
    const std::vector<int> &size, const std::vector<int> &downsample_strides,
    std::vector<NanoDetPlusCenterPoint> *center_points) {
  // size: tuple of input (width, height), e.g (320, 320)
  // downsample_strides: downsample strides in NanoDet and
  // NanoDet-Plus, e.g (8, 16, 32, 64)
  const int width = size[0];
  const int height = size[1];
  for (const auto &ds : downsample_strides) {
    int num_grid_w = width / ds;
    int num_grid_h = height / ds;
    for (int g1 = 0; g1 < num_grid_h; ++g1) {
      for (int g0 = 0; g0 < num_grid_w; ++g0) {
        (*center_points).emplace_back(NanoDetPlusCenterPoint{g0, g1, ds});
      }
    }
  }
}

void WrapAndResize(Mat *mat, std::vector<int> size, std::vector<float> color,
                   bool keep_ratio = false) {
  // Reference: nanodet/data/transform/warp.py#L139
  // size: tuple of input (width, height)
  // The default value of `keep_ratio` is `false` in
  // `config/nanodet-plus-m-1.5x_320.yml` for both
  // train and val processes. So, we just let this
  // option default `false` according to the official
  // implementation in NanoDet and NanoDet-Plus.
  // Note, this function will apply a normal resize
  // operation to input Mat if the keep_ratio option
  // is false and the behavior will be the same as
  // yolov5's letterbox if keep_ratio is true.

  // with keep_ratio = false (default)
  if (!keep_ratio) {
    int resize_h = size[1];
    int resize_w = size[0];
    if (resize_h != mat->Height() || resize_w != mat->Width()) {
      Resize::Run(mat, resize_w, resize_h);
    }
    return;
  }
  // with keep_ratio = true, same as yolov5's letterbox
  float r = std::min(size[1] * 1.0f / static_cast<float>(mat->Height()),
                     size[0] * 1.0f / static_cast<float>(mat->Width()));

  int resize_h = int(round(static_cast<float>(mat->Height()) * r));
  int resize_w = int(round(static_cast<float>(mat->Width()) * r));

  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Resize::Run(mat, resize_w, resize_h);
  }

  int pad_w = size[0] - resize_w;
  int pad_h = size[1] - resize_h;
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

void GFLRegression(const float *logits, size_t reg_num, float *offset) {
  // Hint: reg_num = reg_max + 1
  FDASSERT(((nullptr != logits) && (reg_num != 0)),
           "NanoDetPlus: logits is nullptr or reg_num is 0 in GFLRegression.");
  // softmax
  float total_exp = 0.f;
  std::vector<float> softmax_probs(reg_num);
  for (size_t i = 0; i < reg_num; ++i) {
    softmax_probs[i] = std::exp(logits[i]);
    total_exp += softmax_probs[i];
  }
  for (size_t i = 0; i < reg_num; ++i) {
    softmax_probs[i] = softmax_probs[i] / total_exp;
  }
  // gfl regression -> offset
  for (size_t i = 0; i < reg_num; ++i) {
    (*offset) += static_cast<float>(i) * softmax_probs[i];
  }
}

NanoDetPlus::NanoDetPlus(const std::string &model_file,
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

bool NanoDetPlus::Initialize() {
  // parameters for preprocess
  size = {320, 320};
  padding_value = {0.0f, 0.0f, 0.0f};
  keep_ratio = false;
  downsample_strides = {8, 16, 32, 64};
  max_wh = 4096.0f;
  reg_max = 7;

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

bool NanoDetPlus::Preprocess(
    Mat *mat, FDTensor *output,
    std::map<std::string, std::array<float, 2>> *im_info) {
  // NanoDet-Plus preprocess steps
  // 1. WrapAndResize
  // 2. HWC->CHW
  // 3. Normalize or Convert (keep BGR order)
  WrapAndResize(mat, size, padding_value, keep_ratio);
  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {static_cast<float>(mat->Height()),
                                static_cast<float>(mat->Width())};

  // Compute `result = mat * alpha + beta` directly by channel
  // Reference: /config/nanodet-plus-m-1.5x_320.yml#L89
  // from mean: [103.53, 116.28, 123.675], std: [57.375, 57.12, 58.395]
  // x' = (x - mean) / std to x'= x * alpha + beta.
  // e.g alpha[0] = 0.017429f = 1.0f / 57.375f
  // e.g beta[0] = -103.53f * 0.0174291f
  std::vector<float> alpha = {0.017429f, 0.017507f, 0.017125f};
  std::vector<float> beta = {-103.53f * 0.0174291f, -116.28f * 0.0175070f,
                             -123.675f * 0.0171247f}; // BGR order
  Convert::Run(mat, alpha, beta);

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");
  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1); // reshape to n, c, h, w
  return true;
}

bool NanoDetPlus::Postprocess(
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
  // generate center points with dowmsample strides
  std::vector<NanoDetPlusCenterPoint> center_points;
  GenerateNanoDetPlusCenterPoints(size, downsample_strides, &center_points);

  // infer_result shape might look like (1,2125,112)
  const int num_cls_reg = infer_result.shape[2];           // e.g 112
  const int num_classes = num_cls_reg - (reg_max + 1) * 4; // e.g 80
  float *data = static_cast<float *>(infer_result.Data());
  for (size_t i = 0; i < infer_result.shape[1]; ++i) {
    float *scores = data + i * num_cls_reg;
    float *max_class_score = std::max_element(scores, scores + num_classes);
    float confidence = (*max_class_score);
    // filter boxes by conf_threshold
    if (confidence <= conf_threshold) {
      continue;
    }
    int32_t label_id = std::distance(scores, max_class_score);
    // fetch i-th center point
    float grid0 = static_cast<float>(center_points.at(i).grid0);
    float grid1 = static_cast<float>(center_points.at(i).grid1);
    float downsample_stride = static_cast<float>(center_points.at(i).stride);
    // apply gfl regression to get offsets (l,t,r,b)
    float *logits = data + i * num_cls_reg + num_classes; // 32|44...
    std::vector<float> offsets(4);
    for (size_t j = 0; j < 4; ++j) {
      GFLRegression(logits + j * (reg_max + 1), reg_max + 1, &offsets[j]);
    }
    // convert from offsets to [x1, y1, x2, y2]
    float l = offsets[0]; // left
    float t = offsets[1]; // top
    float r = offsets[2]; // right
    float b = offsets[3]; // bottom

    float x1 = (grid0 - l) * downsample_stride; // cx - l x1
    float y1 = (grid1 - t) * downsample_stride; // cy - t y1
    float x2 = (grid0 + r) * downsample_stride; // cx + r x2
    float y2 = (grid1 + b) * downsample_stride; // cy + b y2

    result->boxes.emplace_back(
        std::array<float, 4>{x1 + label_id * max_wh, y1 + label_id * max_wh,
                             x2 + label_id * max_wh, y2 + label_id * max_wh});
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
  // without keep_ratio
  if (!keep_ratio) {
    // x' = (x / out_w) * ipt_w = x / (out_w / ipt_w)
    // y' = (y / out_h) * ipt_h = y / (out_h / ipt_h)
    float r_w = out_w / ipt_w;
    float r_h = out_h / ipt_h;
    for (size_t i = 0; i < result->boxes.size(); ++i) {
      int32_t label_id = (result->label_ids)[i];
      // clip box
      result->boxes[i][0] = result->boxes[i][0] - max_wh * label_id;
      result->boxes[i][1] = result->boxes[i][1] - max_wh * label_id;
      result->boxes[i][2] = result->boxes[i][2] - max_wh * label_id;
      result->boxes[i][3] = result->boxes[i][3] - max_wh * label_id;
      result->boxes[i][0] = std::max(result->boxes[i][0] / r_w, 0.0f);
      result->boxes[i][1] = std::max(result->boxes[i][1] / r_h, 0.0f);
      result->boxes[i][2] = std::max(result->boxes[i][2] / r_w, 0.0f);
      result->boxes[i][3] = std::max(result->boxes[i][3] / r_h, 0.0f);
      result->boxes[i][0] = std::min(result->boxes[i][0], ipt_w - 1.0f);
      result->boxes[i][1] = std::min(result->boxes[i][1], ipt_h - 1.0f);
      result->boxes[i][2] = std::min(result->boxes[i][2], ipt_w - 1.0f);
      result->boxes[i][3] = std::min(result->boxes[i][3], ipt_h - 1.0f);
    }
    return true;
  }
  // with keep_ratio
  float r = std::min(out_h / ipt_h, out_w / ipt_w);
  float pad_h = (out_h - ipt_h * r) / 2;
  float pad_w = (out_w - ipt_w * r) / 2;
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    int32_t label_id = (result->label_ids)[i];
    // clip box
    result->boxes[i][0] = result->boxes[i][0] - max_wh * label_id;
    result->boxes[i][1] = result->boxes[i][1] - max_wh * label_id;
    result->boxes[i][2] = result->boxes[i][2] - max_wh * label_id;
    result->boxes[i][3] = result->boxes[i][3] - max_wh * label_id;
    result->boxes[i][0] = std::max((result->boxes[i][0] - pad_w) / r, 0.0f);
    result->boxes[i][1] = std::max((result->boxes[i][1] - pad_h) / r, 0.0f);
    result->boxes[i][2] = std::max((result->boxes[i][2] - pad_w) / r, 0.0f);
    result->boxes[i][3] = std::max((result->boxes[i][3] - pad_h) / r, 0.0f);
    result->boxes[i][0] = std::min(result->boxes[i][0], ipt_w - 1.0f);
    result->boxes[i][1] = std::min(result->boxes[i][1], ipt_h - 1.0f);
    result->boxes[i][2] = std::min(result->boxes[i][2], ipt_w - 1.0f);
    result->boxes[i][3] = std::min(result->boxes[i][3], ipt_h - 1.0f);
  }
  return true;
}

bool NanoDetPlus::Predict(cv::Mat *im, DetectionResult *result,
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

} // namespace detection
} // namespace vision
} // namespace ultra_infer
