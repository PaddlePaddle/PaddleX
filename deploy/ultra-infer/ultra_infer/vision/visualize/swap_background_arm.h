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

#include "opencv2/imgproc/imgproc.hpp"
#include "ultra_infer/vision/common/result.h"

namespace ultra_infer {
namespace vision {

cv::Mat SwapBackgroundNEON(const cv::Mat &im, const cv::Mat &background,
                           const MattingResult &result,
                           bool remove_small_connected_area = false);

cv::Mat SwapBackgroundNEON(const cv::Mat &im, const cv::Mat &background,
                           const SegmentationResult &result,
                           int background_label);

} // namespace vision
} // namespace ultra_infer
