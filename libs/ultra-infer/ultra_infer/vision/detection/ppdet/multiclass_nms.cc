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

#include "ultra_infer/vision/detection/ppdet/multiclass_nms.h"
#include "ultra_infer/core/fd_tensor.h"
#include "ultra_infer/utils/utils.h"
#include <algorithm>

namespace ultra_infer {
namespace vision {
namespace detection {
template <class T>
bool SortScorePairDescend(const std::pair<float, T> &pair1,
                          const std::pair<float, T> &pair2) {
  return pair1.first > pair2.first;
}

void GetMaxScoreIndex(const float *scores, const int &score_size,
                      const float &threshold, const int &top_k,
                      std::vector<std::pair<float, int>> *sorted_indices) {
  for (size_t i = 0; i < score_size; ++i) {
    if (scores[i] > threshold) {
      sorted_indices->push_back(std::make_pair(scores[i], i));
    }
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices->begin(), sorted_indices->end(),
                   SortScorePairDescend<int>);
  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < static_cast<int>(sorted_indices->size())) {
    sorted_indices->resize(top_k);
  }
}

float BBoxArea(const float *box, const bool &normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return 0.f;
  } else {
    const float w = box[2] - box[0];
    const float h = box[3] - box[1];
    if (normalized) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}

float JaccardOverlap(const float *box1, const float *box2,
                     const bool &normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return 0.f;
  } else {
    const float inter_xmin = std::max(box1[0], box2[0]);
    const float inter_ymin = std::max(box1[1], box2[1]);
    const float inter_xmax = std::min(box1[2], box2[2]);
    const float inter_ymax = std::min(box1[3], box2[3]);
    float norm = normalized ? 0.0f : 1.0f;
    float inter_w = inter_xmax - inter_xmin + norm;
    float inter_h = inter_ymax - inter_ymin + norm;
    const float inter_area = inter_w * inter_h;
    const float bbox1_area = BBoxArea(box1, normalized);
    const float bbox2_area = BBoxArea(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

void PaddleMultiClassNMS::FastNMS(const float *boxes, const float *scores,
                                  const int &num_boxes,
                                  std::vector<int> *keep_indices) {
  std::vector<std::pair<float, int>> sorted_indices;
  GetMaxScoreIndex(scores, num_boxes, score_threshold, nms_top_k,
                   &sorted_indices);

  float adaptive_threshold = nms_threshold;
  while (sorted_indices.size() != 0) {
    const int idx = sorted_indices.front().second;
    bool keep = true;
    for (size_t k = 0; k < keep_indices->size(); ++k) {
      if (!keep) {
        break;
      }
      const int kept_idx = (*keep_indices)[k];
      float overlap =
          JaccardOverlap(boxes + idx * 4, boxes + kept_idx * 4, normalized);
      keep = overlap <= adaptive_threshold;
    }
    if (keep) {
      keep_indices->push_back(idx);
    }
    sorted_indices.erase(sorted_indices.begin());
    if (keep && nms_eta<1.0 & adaptive_threshold> 0.5) {
      adaptive_threshold *= nms_eta;
    }
  }
}

int PaddleMultiClassNMS::NMSForEachSample(
    const float *boxes, const float *scores, int num_boxes, int num_classes,
    std::map<int, std::vector<int>> *keep_indices) {
  for (int i = 0; i < num_classes; ++i) {
    if (i == background_label) {
      continue;
    }
    const float *score_for_class_i = scores + i * num_boxes;
    FastNMS(boxes, score_for_class_i, num_boxes, &((*keep_indices)[i]));
  }
  int num_det = 0;
  for (auto iter = keep_indices->begin(); iter != keep_indices->end(); ++iter) {
    num_det += iter->second.size();
  }

  if (keep_top_k > -1 && num_det > keep_top_k) {
    std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
    for (const auto &it : *keep_indices) {
      int label = it.first;
      const float *current_score = scores + label * num_boxes;
      auto &label_indices = it.second;
      for (size_t j = 0; j < label_indices.size(); ++j) {
        int idx = label_indices[j];
        score_index_pairs.push_back(
            std::make_pair(current_score[idx], std::make_pair(label, idx)));
      }
    }
    std::stable_sort(score_index_pairs.begin(), score_index_pairs.end(),
                     SortScorePairDescend<std::pair<int, int>>);
    score_index_pairs.resize(keep_top_k);

    std::map<int, std::vector<int>> new_indices;
    for (size_t j = 0; j < score_index_pairs.size(); ++j) {
      int label = score_index_pairs[j].second.first;
      int idx = score_index_pairs[j].second.second;
      new_indices[label].push_back(idx);
    }
    new_indices.swap(*keep_indices);
    num_det = keep_top_k;
  }
  return num_det;
}

void PaddleMultiClassNMS::Compute(const float *boxes_data,
                                  const float *scores_data,
                                  const std::vector<int64_t> &boxes_dim,
                                  const std::vector<int64_t> &scores_dim) {
  int score_size = scores_dim.size();

  int64_t batch_size = scores_dim[0];
  int64_t box_dim = boxes_dim[2];
  int64_t out_dim = box_dim + 2;

  int num_nmsed_out = 0;
  FDASSERT(score_size == 3,
           "Require rank of input scores be 3, but now it's %d.", score_size);
  FDASSERT(boxes_dim[2] == 4,
           "Require the 3-dimension of input boxes be 4, but now it's %lld.",
           box_dim);
  out_num_rois_data.resize(batch_size);

  std::vector<std::map<int, std::vector<int>>> all_indices;
  for (size_t i = 0; i < batch_size; ++i) {
    std::map<int, std::vector<int>> indices; // indices kept for each class
    const float *current_boxes_ptr =
        boxes_data + i * boxes_dim[1] * boxes_dim[2];
    const float *current_scores_ptr =
        scores_data + i * scores_dim[1] * scores_dim[2];
    int num = NMSForEachSample(current_boxes_ptr, current_scores_ptr,
                               boxes_dim[1], scores_dim[1], &indices);
    num_nmsed_out += num;
    out_num_rois_data[i] = num;
    all_indices.emplace_back(indices);
  }
  std::vector<int64_t> out_box_dims = {num_nmsed_out, 6};
  std::vector<int64_t> out_index_dims = {num_nmsed_out, 1};
  if (num_nmsed_out == 0) {
    for (size_t i = 0; i < batch_size; ++i) {
      out_num_rois_data[i] = 0;
    }
    return;
  }
  out_box_data.resize(num_nmsed_out * 6);
  out_index_data.resize(num_nmsed_out);

  int count = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    const float *current_boxes_ptr =
        boxes_data + i * boxes_dim[1] * boxes_dim[2];
    const float *current_scores_ptr =
        scores_data + i * scores_dim[1] * scores_dim[2];
    for (const auto &it : all_indices[i]) {
      int label = it.first;
      const auto &indices = it.second;
      const float *current_scores_class_ptr =
          current_scores_ptr + label * scores_dim[2];
      for (size_t j = 0; j < indices.size(); ++j) {
        int start = count * 6;
        out_box_data[start] = label;
        out_box_data[start + 1] = current_scores_class_ptr[indices[j]];

        out_box_data[start + 2] = current_boxes_ptr[indices[j] * 4];
        out_box_data[start + 3] = current_boxes_ptr[indices[j] * 4 + 1];
        out_box_data[start + 4] = current_boxes_ptr[indices[j] * 4 + 2];

        out_box_data[start + 5] = current_boxes_ptr[indices[j] * 4 + 3];
        out_index_data[count] = i * boxes_dim[1] + indices[j];
        count += 1;
      }
    }
  }
}
} // namespace detection
} // namespace vision
} // namespace ultra_infer
