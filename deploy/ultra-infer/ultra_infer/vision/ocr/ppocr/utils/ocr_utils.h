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

#include "ultra_infer/core/fd_tensor.h"
#include "ultra_infer/utils/utils.h"
#include "ultra_infer/vision/common/result.h"
#include <set>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace ultra_infer {
namespace vision {
namespace ocr {

ULTRAINFER_DECL cv::Mat GetRotateCropImage(const cv::Mat &srcimage,
                                           const std::array<int, 8> &box);

ULTRAINFER_DECL void SortBoxes(std::vector<std::array<int, 8>> *boxes);

ULTRAINFER_DECL std::vector<int> ArgSort(const std::vector<float> &array);

ULTRAINFER_DECL std::vector<float> Softmax(std::vector<float> &src);

ULTRAINFER_DECL std::vector<int> Xyxyxyxy2Xyxy(std::array<int, 8> &box);

ULTRAINFER_DECL float Dis(std::vector<int> &box1, std::vector<int> &box2);

ULTRAINFER_DECL float Iou(std::vector<int> &box1, std::vector<int> &box2);

ULTRAINFER_DECL bool ComparisonDis(const std::vector<float> &dis1,
                                   const std::vector<float> &dis2);
} // namespace ocr
} // namespace vision
} // namespace ultra_infer
