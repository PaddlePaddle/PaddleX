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

#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace utils {

void SortDetectionResult(FaceDetectionResult *result) {
  // sort face detection results with landmarks or not.
  if (result->boxes.size() == 0) {
    return;
  }
  int landmarks_per_face = result->landmarks_per_face;
  if (landmarks_per_face > 0) {
    FDASSERT(
        (result->landmarks.size() == result->boxes.size() * landmarks_per_face),
        "The size of landmarks != boxes.size * landmarks_per_face.");
  }

  // argsort for scores.
  std::vector<size_t> indices;
  indices.resize(result->boxes.size());
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    indices[i] = i;
  }
  std::vector<float> &scores = result->scores;
  std::sort(indices.begin(), indices.end(),
            [&scores](size_t a, size_t b) { return scores[a] > scores[b]; });

  // reorder boxes, scores, landmarks (if have).
  FaceDetectionResult backup(*result);
  result->Clear();
  // don't forget to reset the landmarks_per_face
  // before apply Reserve method.
  result->landmarks_per_face = landmarks_per_face;
  result->Reserve(indices.size());
  if (landmarks_per_face > 0) {
    for (size_t i = 0; i < indices.size(); ++i) {
      result->boxes.emplace_back(backup.boxes[indices[i]]);
      result->scores.push_back(backup.scores[indices[i]]);
      for (size_t j = 0; j < landmarks_per_face; ++j) {
        result->landmarks.emplace_back(
            backup.landmarks[indices[i] * landmarks_per_face + j]);
      }
    }
  } else {
    for (size_t i = 0; i < indices.size(); ++i) {
      result->boxes.emplace_back(backup.boxes[indices[i]]);
      result->scores.push_back(backup.scores[indices[i]]);
    }
  }
}

} // namespace utils
} // namespace vision
} // namespace ultra_infer
