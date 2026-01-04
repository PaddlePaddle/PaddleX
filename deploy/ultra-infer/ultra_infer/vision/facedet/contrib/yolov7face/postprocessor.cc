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

#include "ultra_infer/vision/facedet/contrib/yolov7face/postprocessor.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace facedet {

Yolov7FacePostprocessor::Yolov7FacePostprocessor() {
  conf_threshold_ = 0.5;
  nms_threshold_ = 0.45;
  landmarks_per_face_ = 5;
}

bool Yolov7FacePostprocessor::Run(
    const std::vector<FDTensor> &infer_result,
    std::vector<FaceDetectionResult> *results,
    const std::vector<std::map<std::string, std::array<float, 2>>> &ims_info) {
  int batch = infer_result[0].shape[0];

  results->resize(batch);

  for (size_t bs = 0; bs < batch; ++bs) {
    (*results)[bs].Clear();
    // must be setup landmarks_per_face before reserve
    (*results)[bs].landmarks_per_face = landmarks_per_face_;
    (*results)[bs].Reserve(infer_result[0].shape[1]);
    if (infer_result[0].dtype != FDDataType::FP32) {
      FDERROR << "Only support post process with float32 data." << std::endl;
      return false;
    }
    const float *data =
        reinterpret_cast<const float *>(infer_result[0].Data()) +
        bs * infer_result[0].shape[1] * infer_result[0].shape[2];
    for (size_t i = 0; i < infer_result[0].shape[1]; ++i) {
      int s = i * infer_result[0].shape[2];
      float confidence = data[s + 4];
      const float *reg_cls_ptr = data + s;
      const float *class_score = data + s + 5;
      confidence *= (*class_score);
      // filter boxes by conf_threshold
      if (confidence <= conf_threshold_) {
        continue;
      }
      float x = reg_cls_ptr[0];
      float y = reg_cls_ptr[1];
      float w = reg_cls_ptr[2];
      float h = reg_cls_ptr[3];

      // convert from [x, y, w, h] to [x1, y1, x2, y2]
      (*results)[bs].boxes.emplace_back(std::array<float, 4>{
          (x - w / 2.f), (y - h / 2.f), (x + w / 2.f), (y + h / 2.f)});
      (*results)[bs].scores.push_back(confidence);

      // decode landmarks (default 5 landmarks)
      if (landmarks_per_face_ > 0) {
        float *landmarks_ptr = const_cast<float *>(reg_cls_ptr + 6);
        for (size_t j = 0; j < landmarks_per_face_ * 3; j += 3) {
          (*results)[bs].landmarks.emplace_back(
              std::array<float, 2>{landmarks_ptr[j], landmarks_ptr[j + 1]});
        }
      }
    }

    if ((*results)[bs].boxes.size() == 0) {
      return true;
    }

    utils::NMS(&((*results)[bs]), nms_threshold_);

    // scale the boxes to the origin image shape
    auto iter_out = ims_info[bs].find("output_shape");
    auto iter_ipt = ims_info[bs].find("input_shape");
    FDASSERT(iter_out != ims_info[bs].end() && iter_ipt != ims_info[bs].end(),
             "Cannot find input_shape or output_shape from im_info.");
    float out_h = iter_out->second[0];
    float out_w = iter_out->second[1];
    float ipt_h = iter_ipt->second[0];
    float ipt_w = iter_ipt->second[1];
    float scale = std::min(out_h / ipt_h, out_w / ipt_w);
    float pad_h = (out_h - ipt_h * scale) / 2;
    float pad_w = (out_w - ipt_w * scale) / 2;
    for (size_t i = 0; i < (*results)[bs].boxes.size(); ++i) {
      // clip box
      (*results)[bs].boxes[i][0] =
          std::max(((*results)[bs].boxes[i][0] - pad_w) / scale, 0.0f);
      (*results)[bs].boxes[i][1] =
          std::max(((*results)[bs].boxes[i][1] - pad_h) / scale, 0.0f);
      (*results)[bs].boxes[i][2] =
          std::max(((*results)[bs].boxes[i][2] - pad_w) / scale, 0.0f);
      (*results)[bs].boxes[i][3] =
          std::max(((*results)[bs].boxes[i][3] - pad_h) / scale, 0.0f);
      (*results)[bs].boxes[i][0] =
          std::min((*results)[bs].boxes[i][0], ipt_w - 1.0f);
      (*results)[bs].boxes[i][1] =
          std::min((*results)[bs].boxes[i][1], ipt_h - 1.0f);
      (*results)[bs].boxes[i][2] =
          std::min((*results)[bs].boxes[i][2], ipt_w - 1.0f);
      (*results)[bs].boxes[i][3] =
          std::min((*results)[bs].boxes[i][3], ipt_h - 1.0f);
    }

    // scale and clip landmarks
    for (size_t i = 0; i < (*results)[bs].landmarks.size(); ++i) {
      (*results)[bs].landmarks[i][0] =
          std::max(((*results)[bs].landmarks[i][0] - pad_w) / scale, 0.0f);
      (*results)[bs].landmarks[i][1] =
          std::max(((*results)[bs].landmarks[i][1] - pad_h) / scale, 0.0f);
      (*results)[bs].landmarks[i][0] =
          std::min((*results)[bs].landmarks[i][0], ipt_w - 1.0f);
      (*results)[bs].landmarks[i][1] =
          std::min((*results)[bs].landmarks[i][1], ipt_h - 1.0f);
    }
  }
  return true;
}

} // namespace facedet
} // namespace vision
} // namespace ultra_infer
