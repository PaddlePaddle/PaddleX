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

#pragma once
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace keypointdetection {

cv::Point2f Get3dPoint(const cv::Point2f &a, const cv::Point2f &b);

std::vector<float> GetDir(const float src_point_x, const float src_point_y,
                          const float rot_rad);

void GetAffineTransform(const std::vector<float> &center,
                        const std::vector<float> &scale, const float rot,
                        const std::vector<int> &output_size, cv::Mat *trans,
                        const int inv);

void AffineTransform(const float pt_x, const float pt_y, const cv::Mat &trans,
                     std::vector<float> *preds, const int p);

void TransformPreds(std::vector<float> &coords,
                    const std::vector<float> &center,
                    const std::vector<float> &scale,
                    const std::vector<int> &output_size,
                    const std::vector<int> &dim,
                    std::vector<float> *target_coords);

void GetFinalPredictions(const std::vector<float> &heatmap,
                         const std::vector<int> &dim,
                         const std::vector<int64_t> &idxout,
                         const std::vector<float> &center,
                         const std::vector<float> scale,
                         std::vector<float> *preds, const bool DARK);

} // namespace keypointdetection
} // namespace vision
} // namespace ultra_infer
