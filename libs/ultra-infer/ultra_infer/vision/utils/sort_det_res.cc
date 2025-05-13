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

void Merge(DetectionResult *result, size_t low, size_t mid, size_t high) {
  std::vector<std::array<float, 4>> &boxes = result->boxes;
  std::vector<float> &scores = result->scores;
  std::vector<int32_t> &label_ids = result->label_ids;
  std::vector<std::array<float, 4>> temp_boxes(boxes);
  std::vector<float> temp_scores(scores);
  std::vector<int32_t> temp_label_ids(label_ids);
  size_t i = low;
  size_t j = mid + 1;
  size_t k = i;
  // TODO(qiuyanjun): add masks process
  for (; i <= mid && j <= high; k++) {
    if (temp_scores[i] >= temp_scores[j]) {
      scores[k] = temp_scores[i];
      label_ids[k] = temp_label_ids[i];
      boxes[k] = temp_boxes[i];
      i++;
    } else {
      scores[k] = temp_scores[j];
      label_ids[k] = temp_label_ids[j];
      boxes[k] = temp_boxes[j];
      j++;
    }
  }
  while (i <= mid) {
    scores[k] = temp_scores[i];
    label_ids[k] = temp_label_ids[i];
    boxes[k] = temp_boxes[i];
    k++;
    i++;
  }
  while (j <= high) {
    scores[k] = temp_scores[j];
    label_ids[k] = temp_label_ids[j];
    boxes[k] = temp_boxes[j];
    k++;
    j++;
  }
}

void MergeSort(DetectionResult *result, size_t low, size_t high) {
  if (low < high) {
    size_t mid = (high - low) / 2 + low;
    MergeSort(result, low, mid);
    MergeSort(result, mid + 1, high);
    Merge(result, low, mid, high);
  }
}

void SortDetectionResult(DetectionResult *result) {
  size_t low = 0;
  size_t high = result->scores.size();
  if (high == 0) {
    return;
  }
  high = high - 1;
  MergeSort(result, low, high);
}

template <typename T>
bool LexSortByXYCompare(const std::array<T, 4> &box_a,
                        const std::array<T, 4> &box_b) {
  // WARN: The status should be false if (a==b).
  // https://blog.csdn.net/xxxwrq/article/details/83080640
  auto is_equal = [](const T &a, const T &b) -> bool {
    return std::abs(a - b) < 1e-6f;
  };
  const T &x0_a = box_a[0];
  const T &y0_a = box_a[1];
  const T &x0_b = box_b[0];
  const T &y0_b = box_b[1];
  if (is_equal(x0_a, x0_b)) {
    return is_equal(y0_a, y0_b) ? false : y0_a > y0_b;
  }
  return x0_a > x0_b;
}

// Only for int dtype
template <>
bool LexSortByXYCompare(const std::array<int, 4> &box_a,
                        const std::array<int, 4> &box_b) {
  const int &x0_a = box_a[0];
  const int &y0_a = box_a[1];
  const int &x0_b = box_b[0];
  const int &y0_b = box_b[1];
  if (x0_a == x0_b) {
    return y0_a == y0_b ? false : y0_a > y0_b;
  }
  return x0_a > x0_b;
}

void ReorderDetectionResultByIndices(DetectionResult *result,
                                     const std::vector<size_t> &indices) {
  // reorder boxes, scores, label_ids, masks
  DetectionResult backup = (*result);
  const bool contain_masks = backup.contain_masks;
  const int boxes_num = backup.boxes.size();
  result->Clear();
  result->Resize(boxes_num);
  // boxes, scores, labels_ids
  for (int i = 0; i < boxes_num; ++i) {
    result->boxes[i] = backup.boxes[indices[i]];
    result->scores[i] = backup.scores[indices[i]];
    result->label_ids[i] = backup.label_ids[indices[i]];
  }
  if (contain_masks) {
    result->contain_masks = true;
    for (int i = 0; i < boxes_num; ++i) {
      const auto &shape = backup.masks[indices[i]].shape;
      const int mask_numel = shape[0] * shape[1];
      result->masks[i].shape = shape;
      result->masks[i].Resize(mask_numel);
      std::memcpy(result->masks[i].Data(), backup.masks[indices[i]].Data(),
                  mask_numel * sizeof(uint8_t));
    }
  }
}

void LexSortDetectionResultByXY(DetectionResult *result) {
  if (result->boxes.empty()) {
    return;
  }
  std::vector<size_t> indices;
  indices.resize(result->boxes.size());
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    indices[i] = i;
  }
  // lex sort by x(w) then y(h)
  auto &boxes = result->boxes;
  std::sort(indices.begin(), indices.end(), [&boxes](size_t a, size_t b) {
    return LexSortByXYCompare(boxes[a], boxes[b]);
  });
  ReorderDetectionResultByIndices(result, indices);
}

void LexSortOCRDetResultByXY(std::vector<std::array<int, 8>> *result) {
  if (result->empty()) {
    return;
  }
  std::vector<size_t> indices;
  indices.resize(result->size());
  std::vector<std::array<int, 4>> boxes;
  boxes.resize(result->size());
  for (size_t i = 0; i < result->size(); ++i) {
    indices[i] = i;
    // 4 points to 2 points for LexSort
    boxes[i] = {(*result)[i][0], (*result)[i][1], (*result)[i][6],
                (*result)[i][7]};
  }
  // lex sort by x(w) then y(h)
  std::sort(indices.begin(), indices.end(), [&boxes](size_t a, size_t b) {
    return LexSortByXYCompare(boxes[a], boxes[b]);
  });
  // reorder boxes
  std::vector<std::array<int, 8>> backup = (*result);
  const int boxes_num = backup.size();
  result->clear();
  result->resize(boxes_num);
  // boxes
  for (int i = 0; i < boxes_num; ++i) {
    (*result)[i] = backup[indices[i]];
  }
}

} // namespace utils
} // namespace vision
} // namespace ultra_infer
