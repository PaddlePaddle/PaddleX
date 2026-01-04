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

#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace utils {

// The implementation refers to
// https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/cpp/src/utils.cc
void NMS(DetectionResult *result, float iou_threshold,
         std::vector<int> *index) {
  // get sorted score indices
  std::vector<int> sorted_indices;
  if (index != nullptr) {
    std::map<float, int, std::greater<float>> score_map;
    for (size_t i = 0; i < result->scores.size(); ++i) {
      score_map.insert(std::pair<float, int>(result->scores[i], i));
    }
    for (auto iter : score_map) {
      sorted_indices.push_back(iter.second);
    }
  }
  utils::SortDetectionResult(result);

  std::vector<float> area_of_boxes(result->boxes.size());
  std::vector<int> suppressed(result->boxes.size(), 0);
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    area_of_boxes[i] = (result->boxes[i][2] - result->boxes[i][0]) *
                       (result->boxes[i][3] - result->boxes[i][1]);
  }

  for (size_t i = 0; i < result->boxes.size(); ++i) {
    if (suppressed[i] == 1) {
      continue;
    }
    for (size_t j = i + 1; j < result->boxes.size(); ++j) {
      if (suppressed[j] == 1) {
        continue;
      }
      float xmin = std::max(result->boxes[i][0], result->boxes[j][0]);
      float ymin = std::max(result->boxes[i][1], result->boxes[j][1]);
      float xmax = std::min(result->boxes[i][2], result->boxes[j][2]);
      float ymax = std::min(result->boxes[i][3], result->boxes[j][3]);
      float overlap_w = std::max(0.0f, xmax - xmin);
      float overlap_h = std::max(0.0f, ymax - ymin);
      float overlap_area = overlap_w * overlap_h;
      float overlap_ratio =
          overlap_area / (area_of_boxes[i] + area_of_boxes[j] - overlap_area);
      if (overlap_ratio > iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
  DetectionResult backup(*result);
  result->Clear();
  result->Reserve(suppressed.size());
  for (size_t i = 0; i < suppressed.size(); ++i) {
    if (suppressed[i] == 1) {
      continue;
    }
    result->boxes.emplace_back(backup.boxes[i]);
    result->scores.push_back(backup.scores[i]);
    result->label_ids.push_back(backup.label_ids[i]);
    if (index != nullptr) {
      index->push_back(sorted_indices[i]);
    }
  }
}

void NMS(FaceDetectionResult *result, float iou_threshold) {
  utils::SortDetectionResult(result);

  std::vector<float> area_of_boxes(result->boxes.size());
  std::vector<int> suppressed(result->boxes.size(), 0);
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    area_of_boxes[i] = (result->boxes[i][2] - result->boxes[i][0]) *
                       (result->boxes[i][3] - result->boxes[i][1]);
  }

  for (size_t i = 0; i < result->boxes.size(); ++i) {
    if (suppressed[i] == 1) {
      continue;
    }
    for (size_t j = i + 1; j < result->boxes.size(); ++j) {
      if (suppressed[j] == 1) {
        continue;
      }
      float xmin = std::max(result->boxes[i][0], result->boxes[j][0]);
      float ymin = std::max(result->boxes[i][1], result->boxes[j][1]);
      float xmax = std::min(result->boxes[i][2], result->boxes[j][2]);
      float ymax = std::min(result->boxes[i][3], result->boxes[j][3]);
      float overlap_w = std::max(0.0f, xmax - xmin);
      float overlap_h = std::max(0.0f, ymax - ymin);
      float overlap_area = overlap_w * overlap_h;
      float overlap_ratio =
          overlap_area / (area_of_boxes[i] + area_of_boxes[j] - overlap_area);
      if (overlap_ratio > iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
  FaceDetectionResult backup(*result);
  int landmarks_per_face = result->landmarks_per_face;

  result->Clear();
  // don't forget to reset the landmarks_per_face
  // before apply Reserve method.
  result->landmarks_per_face = landmarks_per_face;
  result->Reserve(suppressed.size());
  for (size_t i = 0; i < suppressed.size(); ++i) {
    if (suppressed[i] == 1) {
      continue;
    }
    result->boxes.emplace_back(backup.boxes[i]);
    result->scores.push_back(backup.scores[i]);
    // landmarks (if have)
    if (result->landmarks_per_face > 0) {
      for (size_t j = 0; j < result->landmarks_per_face; ++j) {
        result->landmarks.emplace_back(
            backup.landmarks[i * result->landmarks_per_face + j]);
      }
    }
  }
}

} // namespace utils
} // namespace vision
} // namespace ultra_infer
