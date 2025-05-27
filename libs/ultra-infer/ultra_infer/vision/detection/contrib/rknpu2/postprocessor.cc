// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "ultra_infer/vision/detection/contrib/rknpu2/postprocessor.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace detection {

RKYOLOPostprocessor::RKYOLOPostprocessor() {}

bool RKYOLOPostprocessor::Run(const std::vector<FDTensor> &tensors,
                              std::vector<DetectionResult> *results) {
  results->resize(tensors[0].shape[0]);
  for (int num = 0; num < tensors[0].shape[0]; ++num) {
    int validCount = 0;
    std::vector<float> filterBoxes;
    std::vector<float> boxesScore;
    std::vector<int> classId;
    for (int i = 0; i < tensors.size(); i++) {
      auto tensor_shape = tensors[i].shape;
      auto skip_num = std::accumulate(tensor_shape.begin(), tensor_shape.end(),
                                      1, std::multiplies<int>());
      int skip_address = num * skip_num;
      int stride = strides_[i];
      int grid_h = height_ / stride;
      int grid_w = width_ / stride;
      int *anchor = &(anchors_.data()[i * 2 * anchor_per_branch_]);
      if (tensors[i].dtype == FDDataType::FP32) {
        validCount = validCount +
                     ProcessFP16((float *)tensors[i].Data() + skip_address,
                                 anchor, grid_h, grid_w, stride, filterBoxes,
                                 boxesScore, classId, conf_threshold_);
      } else {
        FDERROR << "RKYOLO Only Support FP32 Model."
                << "But the result's type is " << Str(tensors[i].dtype)
                << std::endl;
      }
    }

    // no object detect
    if (validCount <= 0) {
      FDINFO << "The number of object detect is 0." << std::endl;
      return true;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
      indexArray.push_back(i);
    }

    QuickSortIndiceInverse(boxesScore, 0, validCount - 1, indexArray);

    if (anchor_per_branch_ == 3) {
      NMS(validCount, filterBoxes, classId, indexArray, nms_threshold_, false);
    } else if (anchor_per_branch_ == 1) {
      NMS(validCount, filterBoxes, classId, indexArray, nms_threshold_, true);
    } else {
      FDERROR << "anchor_per_branch_ only support 3 or 1." << std::endl;
      return false;
    }

    int last_count = 0;
    (*results)[num].Clear();
    (*results)[num].Reserve(validCount);

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
      if (indexArray[i] == -1 || boxesScore[i] < conf_threshold_ ||
          last_count >= obj_num_bbox_max_size) {
        continue;
      }
      int n = indexArray[i];
      float x1 = filterBoxes[n * 4 + 0];
      float y1 = filterBoxes[n * 4 + 1];
      float x2 = x1 + filterBoxes[n * 4 + 2];
      float y2 = y1 + filterBoxes[n * 4 + 3];
      int id = classId[n];
      (*results)[num].boxes.emplace_back(std::array<float, 4>{
          (float)((Clamp(x1, 0, width_) - pad_hw_values_[num][1] / 2) /
                  scale_[num]),
          (float)((Clamp(y1, 0, height_) - pad_hw_values_[num][0] / 2) /
                  scale_[num]),
          (float)((Clamp(x2, 0, width_) - pad_hw_values_[num][1] / 2) /
                  scale_[num]),
          (float)((Clamp(y2, 0, height_) - pad_hw_values_[num][0] / 2) /
                  scale_[0])});
      (*results)[num].label_ids.push_back(id);
      (*results)[num].scores.push_back(boxesScore[i]);
      last_count++;
    }
  }
  return true;
}

int RKYOLOPostprocessor::ProcessFP16(float *input, int *anchor, int grid_h,
                                     int grid_w, int stride,
                                     std::vector<float> &boxes,
                                     std::vector<float> &boxScores,
                                     std::vector<int> &classId,
                                     float threshold) {

  int validCount = 0;
  int grid_len = grid_h * grid_w;
  // float thres_sigmoid = threshold;
  for (int a = 0; a < anchor_per_branch_; a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        float box_confidence =
            input[(prob_box_size_ * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= threshold) {
          int offset = (prob_box_size_ * a) * grid_len + i * grid_w + j;
          float *in_ptr = input + offset;

          float maxClassProbs = in_ptr[5 * grid_len];
          int maxClassId = 0;
          for (int k = 1; k < obj_class_num_; ++k) {
            float prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs) {
              maxClassId = k;
              maxClassProbs = prob;
            }
          }
          float box_conf_f32 = (box_confidence);
          float class_prob_f32 = (maxClassProbs);
          float limit_score = 0;
          if (anchor_per_branch_ == 1) {
            limit_score = class_prob_f32;
          } else {
            limit_score = box_conf_f32 * class_prob_f32;
          }
          if (limit_score > conf_threshold_) {
            float box_x, box_y, box_w, box_h;
            if (anchor_per_branch_ == 1) {
              box_x = *in_ptr;
              box_y = (in_ptr[grid_len]);
              box_w = exp(in_ptr[2 * grid_len]) * stride;
              box_h = exp(in_ptr[3 * grid_len]) * stride;
            } else {
              box_x = *in_ptr * 2.0 - 0.5;
              box_y = (in_ptr[grid_len]) * 2.0 - 0.5;
              box_w = (in_ptr[2 * grid_len]) * 2.0;
              box_h = (in_ptr[3 * grid_len]) * 2.0;
              box_w *= box_w;
              box_h *= box_h;
            }
            box_x = (box_x + j) * (float)stride;
            box_y = (box_y + i) * (float)stride;
            box_w *= (float)anchor[a * 2];
            box_h *= (float)anchor[a * 2 + 1];
            box_x -= (box_w / 2.0);
            box_y -= (box_h / 2.0);

            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
            boxScores.push_back(box_conf_f32 * class_prob_f32);
            classId.push_back(maxClassId);
            validCount++;
          }
        }
      }
    }
  }
  return validCount;
}

int RKYOLOPostprocessor::QuickSortIndiceInverse(std::vector<float> &input,
                                                int left, int right,
                                                std::vector<int> &indices) {
  float key;
  int key_index;
  int low = left;
  int high = right;
  if (left < right) {
    key_index = indices[left];
    key = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low] = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high] = input[low];
      indices[high] = indices[low];
    }
    input[low] = key;
    indices[low] = key_index;
    QuickSortIndiceInverse(input, left, low - 1, indices);
    QuickSortIndiceInverse(input, low + 1, right, indices);
  }
  return low;
}

} // namespace detection
} // namespace vision
} // namespace ultra_infer
