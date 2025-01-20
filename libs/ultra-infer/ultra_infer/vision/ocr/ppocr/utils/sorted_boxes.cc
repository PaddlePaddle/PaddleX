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

#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

bool CompareBox(const std::array<int, 8> &result1,
                const std::array<int, 8> &result2) {
  if (result1[1] < result2[1]) {
    return true;
  } else if (result1[1] == result2[1]) {
    return result1[0] < result2[0];
  } else {
    return false;
  }
}

void SortBoxes(std::vector<std::array<int, 8>> *boxes) {
  std::sort(boxes->begin(), boxes->end(), CompareBox);

  if (boxes->size() == 0) {
    return;
  }

  for (int i = 0; i < boxes->size() - 1; i++) {
    for (int j = i; j >= 0; j--) {
      if (std::abs((*boxes)[j + 1][1] - (*boxes)[j][1]) < 10 &&
          ((*boxes)[j + 1][0] < (*boxes)[j][0])) {
        std::swap((*boxes)[i], (*boxes)[i + 1]);
      }
    }
  }
}

std::vector<int> ArgSort(const std::vector<float> &array) {
  const int array_len(array.size());
  std::vector<int> array_index(array_len, 0);
  for (int i = 0; i < array_len; ++i)
    array_index[i] = i;

  std::sort(
      array_index.begin(), array_index.end(),
      [&array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });

  return array_index;
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
