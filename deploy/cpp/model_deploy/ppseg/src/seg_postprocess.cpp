// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "model_deploy/ppseg/include/seg_postprocess.h"

namespace PaddleDeploy {

bool SegPostProcess::Init(const YAML::Node& yaml_config) {
  return true;
}

void SegPostProcess::RestoreResult(
                    const float* ptr,
                    const std::vector<int>& shape,
                    const ShapeInfo& shape_info,
                    SegResult* result) {
  result->score_map.Resize(shape_info.shapes[0]);
  result->label_map.Resize(shape_info.shapes[0]);

  // read result from memory buffer
  // convert to label_map and score_map
  int num_pixels = shape[2] * shape[3];
  std::vector<float> score_map(num_pixels);
  std::vector<float> label_map(num_pixels);
  for (int i = 0; i < num_pixels; ++i) {
    std::vector<float> pixel_score(shape[1]);
    for (int j = 0; j < shape[1]; ++j) {
      pixel_score[j] = *(ptr + i + j * num_pixels);
    }
    int index = std::max_element(pixel_score.begin(),
                    pixel_score.end()) - pixel_score.begin();
    label_map[i] = index;
    score_map[i] = pixel_score[index];
  }

  // recover label map and score map to align origin image
  bool need_recover = false;
  if (shape_info.shapes[0][0] != shape_info.shapes.back()[0] ||
      shape_info.shapes[0][1] != shape_info.shapes.back()[1]) {
    need_recover = true;
  }
  if (need_recover) {
    cv::Mat mask_label(shape[2], shape[3], CV_8UC1, label_map.data());
    cv::Mat mask_score(shape[2], shape[1], CV_32FC1, score_map.data());
    for (int j = shape_info.shapes.size() - 1; j >= 0; --j) {
      if (shape_info.transforms[j] == "Padding") {
          std::vector<int> last_shape = shape_info.shapes[j - 1];
          mask_label = mask_label(cv::Rect(0, 0, last_shape[0], last_shape[1]));
          mask_score = mask_score(cv::Rect(0, 0, last_shape[0], last_shape[1]));
      } else if (shape_info.transforms[j] == "Resize") {
          std::vector<int> last_shape = shape_info.shapes[j - 1];
          cv::resize(mask_label, mask_label,
                    cv::Size(last_shape[0], last_shape[1]),
                    0, 0, cv::INTER_NEAREST);
          cv::resize(mask_score, mask_score,
                    cv::Size(last_shape[0], last_shape[1]),
                    0, 0, cv::INTER_LINEAR);
      }
    }
    result->label_map.data.assign(mask_label.begin<uint8_t>(),
                        mask_label.end<uint8_t>());
    result->score_map.data.assign(mask_score.begin<float>(),
                        mask_score.end<float>());
  } else {
    result->label_map.data.assign(label_map.begin(), label_map.end());
    result->score_map.data.assign(score_map.begin(), score_map.end());
  }
}

bool SegPostProcess::Run(const std::vector<DataBlob>& outputs,
                         const std::vector<ShapeInfo>& shape_infos,
                         std::vector<Result>* results, int thread_num) {
  results->clear();
  int batch_size = shape_infos.size();
  results->resize(batch_size);

  std::vector<int> score_map_shape = outputs[0].shape;
  int score_map_size = std::accumulate(score_map_shape.begin() + 1,
                    score_map_shape.end(), 1, std::multiplies<int>());
  const float* data = reinterpret_cast<const float*>(outputs[0].data.data());

  for (int i = 0; i < batch_size; ++i) {
    (*results)[i].model_type = "seg";
    (*results)[i].seg_result = new SegResult();
    RestoreResult(data + i * score_map_size, score_map_shape,
                shape_infos[i], (*results)[i].seg_result);
  }
  return true;
}

}  //  namespace PaddleDeploy
