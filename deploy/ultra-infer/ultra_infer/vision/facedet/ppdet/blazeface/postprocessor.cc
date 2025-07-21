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

#include "ultra_infer/vision/facedet/ppdet/blazeface/postprocessor.h"
#include "ultra_infer/vision/detection/ppdet/multiclass_nms.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {

namespace vision {

namespace facedet {

BlazeFacePostprocessor::BlazeFacePostprocessor() {
  conf_threshold_ = 0.5;
  nms_threshold_ = 0.3;
}

bool BlazeFacePostprocessor::Run(
    const std::vector<FDTensor> &tensors,
    std::vector<FaceDetectionResult> *results,
    const std::vector<std::map<std::string, std::array<float, 2>>> &ims_info) {
  // Get number of boxes for each input image
  std::vector<int> num_boxes(tensors[1].shape[0]);
  int total_num_boxes = 0;
  if (tensors[1].dtype == FDDataType::INT32) {
    const auto *data = static_cast<const int32_t *>(tensors[1].CpuData());
    for (size_t i = 0; i < tensors[1].shape[0]; ++i) {
      num_boxes[i] = static_cast<int>(data[i]);
      total_num_boxes += num_boxes[i];
    }
  } else if (tensors[1].dtype == FDDataType::INT64) {
    const auto *data = static_cast<const int64_t *>(tensors[1].CpuData());
    for (size_t i = 0; i < tensors[1].shape[0]; ++i) {
      num_boxes[i] = static_cast<int>(data[i]);
    }
  }

  // Special case for TensorRT, it has fixed output shape of NMS
  // So there's invalid boxes in its' output boxes
  int num_output_boxes = static_cast<int>(tensors[0].Shape()[0]);
  bool contain_invalid_boxes = false;
  if (total_num_boxes != num_output_boxes) {
    if (num_output_boxes % num_boxes.size() == 0) {
      contain_invalid_boxes = true;
    } else {
      FDERROR << "Cannot handle the output data for this model, unexpected "
                 "situation."
              << std::endl;
      return false;
    }
  }

  // Get boxes for each input image
  results->resize(num_boxes.size());

  if (tensors[0].shape[0] == 0) {
    // No detected boxes
    return true;
  }

  const auto *box_data = static_cast<const float *>(tensors[0].CpuData());
  int offset = 0;
  for (size_t i = 0; i < num_boxes.size(); ++i) {
    const float *ptr = box_data + offset;
    (*results)[i].Reserve(num_boxes[i]);
    for (size_t j = 0; j < num_boxes[i]; ++j) {
      if (ptr[j * 6 + 1] > conf_threshold_) {
        (*results)[i].scores.push_back(ptr[j * 6 + 1]);
        (*results)[i].boxes.emplace_back(std::array<float, 4>(
            {ptr[j * 6 + 2], ptr[j * 6 + 3], ptr[j * 6 + 4], ptr[j * 6 + 5]}));
      }
    }
    if (contain_invalid_boxes) {
      offset += static_cast<int>(num_output_boxes * 6 / num_boxes.size());
    } else {
      offset += static_cast<int>(num_boxes[i] * 6);
    }
  }
  return true;
}

} // namespace facedet
} // namespace vision
} // namespace ultra_infer
