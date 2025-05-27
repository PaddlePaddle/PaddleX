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

#include "ultra_infer/vision/facedet/contrib/centerface/postprocessor.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace facedet {

CenterFacePostprocessor::CenterFacePostprocessor() {
  conf_threshold_ = 0.5;
  nms_threshold_ = 0.3;
  landmarks_per_face_ = 5;
}

bool CenterFacePostprocessor::Run(
    const std::vector<FDTensor> &infer_result,
    std::vector<FaceDetectionResult> *results,
    const std::vector<std::map<std::string, std::array<float, 2>>> &ims_info) {
  int batch = infer_result[0].shape[0];

  results->resize(batch);
  FDTensor heatmap = infer_result[0];   //(1 1 160 160)
  FDTensor scales = infer_result[1];    //(1 2 160 160)
  FDTensor offsets = infer_result[2];   //(1 2 160 160)
  FDTensor landmarks = infer_result[3]; //(1 10 160 160)
  for (size_t bs = 0; bs < batch; ++bs) {
    (*results)[bs].Clear();
    (*results)[bs].landmarks_per_face = landmarks_per_face_;
    (*results)[bs].Reserve(heatmap.shape[2]);
    if (infer_result[0].dtype != FDDataType::FP32) {
      FDERROR << "Only support post process with float32 data." << std::endl;
      return false;
    }
    int fea_h = heatmap.shape[2];
    int fea_w = heatmap.shape[3];
    int spacial_size = fea_w * fea_h;

    float *heatmap_out = static_cast<float *>(heatmap.Data());

    float *scale0 = static_cast<float *>(scales.Data());
    float *scale1 = scale0 + spacial_size;

    float *offset0 = static_cast<float *>(offsets.Data());
    float *offset1 = offset0 + spacial_size;
    float confidence = 0.f;

    std::vector<int> ids;
    for (int i = 0; i < fea_h; i++) {
      for (int j = 0; j < fea_w; j++) {
        if (heatmap_out[i * fea_w + j] > conf_threshold_) {
          ids.push_back(i);
          ids.push_back(j);
        }
      }
    }

    auto iter_out = ims_info[bs].find("output_shape");
    auto iter_ipt = ims_info[bs].find("input_shape");
    FDASSERT(iter_out != ims_info[bs].end() && iter_ipt != ims_info[bs].end(),
             "Cannot find input_shape or output_shape from im_info.");
    float out_h = iter_out->second[0];
    float out_w = iter_out->second[1];
    float ipt_h = iter_ipt->second[0];
    float ipt_w = iter_ipt->second[1];
    float scale_h = ipt_h / out_h;
    float scale_w = ipt_w / out_w;

    for (int i = 0; i < ids.size() / 2; i++) {
      int id_h = ids[2 * i];
      int id_w = ids[2 * i + 1];
      int index = id_h * fea_w + id_w;
      confidence = heatmap_out[index];

      float s0 = std::exp(scale0[index]) * 4;
      float s1 = std::exp(scale1[index]) * 4;
      float o0 = offset0[index];
      float o1 = offset1[index];

      float x1 = (id_w + o1 + 0.5) * 4 - s1 / 2 > 0.f
                     ? (id_w + o1 + 0.5) * 4 - s1 / 2
                     : 0;
      float y1 = (id_h + o0 + 0.5) * 4 - s0 / 2 > 0
                     ? (id_h + o0 + 0.5) * 4 - s0 / 2
                     : 0;
      float x2 = 0, y2 = 0;
      x1 = x1 < (float)out_w ? x1 : (float)out_w;
      y1 = y1 < (float)out_h ? y1 : (float)out_h;
      x2 = x1 + s1 < (float)out_w ? x1 + s1 : (float)out_w;
      y2 = y1 + s0 < (float)out_h ? y1 + s0 : (float)out_h;

      (*results)[bs].boxes.emplace_back(std::array<float, 4>{x1, y1, x2, y2});
      (*results)[bs].scores.push_back(confidence);
      // decode landmarks (default 5 landmarks)
      if (landmarks_per_face_ > 0) {
        // reference: utils/box_utils.py#L241
        for (size_t j = 0; j < landmarks_per_face_; j++) {
          float *xmap = (float *)landmarks.Data() + (2 * j + 1) * spacial_size;
          float *ymap = (float *)landmarks.Data() + (2 * j) * spacial_size;
          float lx = (x1 + xmap[index] * s1) * scale_w;
          float ly = (y1 + ymap[index] * s0) * scale_h;
          (*results)[bs].landmarks.emplace_back(std::array<float, 2>{lx, ly});
        }
      }
    }

    if ((*results)[bs].boxes.size() == 0) {
      return true;
    }

    utils::NMS(&((*results)[bs]), nms_threshold_);

    for (size_t i = 0; i < (*results)[bs].boxes.size(); ++i) {
      (*results)[bs].boxes[i][0] =
          std::max((*results)[bs].boxes[i][0] * scale_w, 0.0f);
      (*results)[bs].boxes[i][1] =
          std::max((*results)[bs].boxes[i][1] * scale_h, 0.0f);
      (*results)[bs].boxes[i][2] =
          std::max((*results)[bs].boxes[i][2] * scale_w, 0.0f);
      (*results)[bs].boxes[i][3] =
          std::max((*results)[bs].boxes[i][3] * scale_h, 0.0f);
      (*results)[bs].boxes[i][0] =
          std::min((*results)[bs].boxes[i][0], ipt_w - 1.0f);
      (*results)[bs].boxes[i][1] =
          std::min((*results)[bs].boxes[i][1], ipt_h - 1.0f);
      (*results)[bs].boxes[i][2] =
          std::min((*results)[bs].boxes[i][2], ipt_w - 1.0f);
      (*results)[bs].boxes[i][3] =
          std::min((*results)[bs].boxes[i][3], ipt_h - 1.0f);
    }
  }
  return true;
}

} // namespace facedet
} // namespace vision
} // namespace ultra_infer
