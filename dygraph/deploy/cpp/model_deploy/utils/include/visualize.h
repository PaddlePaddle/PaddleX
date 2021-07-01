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

#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "model_deploy/common/include/output_struct.h"

namespace PaddleDeploy {

bool GenerateColorMap(const int &num_classes,
                      std::vector<int> *color_map);

bool Visualize(const cv::Mat& img,
               const DetResult& results,
               cv::Mat* vis_img,
               const int& num_classes = 81);

bool Visualize(const cv::Mat& img,
               const SegResult& result,
               cv::Mat* vis_img,
               const int& num_classes = 81);

}  // namespace PaddleDeploy
