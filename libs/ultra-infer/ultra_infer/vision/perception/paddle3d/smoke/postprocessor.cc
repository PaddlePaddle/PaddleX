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

#include "ultra_infer/vision/perception/paddle3d/smoke/postprocessor.h"

#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace perception {

SmokePostprocessor::SmokePostprocessor() {}

bool SmokePostprocessor::Run(const std::vector<FDTensor> &tensors,
                             std::vector<PerceptionResult> *results) {
  results->resize(1);
  (*results)[0].Clear();
  (*results)[0].Reserve(tensors[0].shape[0]);
  if (tensors[0].dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }
  const float *data = reinterpret_cast<const float *>(tensors[0].Data());
  auto result = &(*results)[0];
  for (int i = 0; i < tensors[0].shape[0] * tensors[0].shape[1]; i += 14) {
    // item 1       :  class
    // item 2       :  observation angle α
    // item 3 ~ 6   :  box2d x1, y1, x2, y2
    // item 7 ~ 9   :  box3d h, w, l
    // item 10 ~ 12 :  box3d bottom center x, y, z
    // item 13      :  box3d yaw angle
    // item 14      :  score
    std::vector<float> vec(data + i, data + i + 14);
    result->scores.push_back(vec[13]);
    result->label_ids.push_back(vec[0]);
    result->boxes.emplace_back(std::array<float, 7>{
        vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8]});
    result->center.emplace_back(std::array<float, 3>{vec[9], vec[10], vec[11]});
    result->observation_angle.push_back(vec[1]);
    result->yaw_angle.push_back(vec[12]);
  }

  result->valid.push_back(true);  // 0 scores
  result->valid.push_back(true);  // 1 label_ids
  result->valid.push_back(true);  // 2 boxes
  result->valid.push_back(true);  // 3 center
  result->valid.push_back(true);  // 4 observation_angle
  result->valid.push_back(true);  // 5 yaw_angle
  result->valid.push_back(false); // 6 velocity

  return true;
}

} // namespace perception
} // namespace vision
} // namespace ultra_infer
