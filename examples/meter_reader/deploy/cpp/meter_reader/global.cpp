// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


#include <iostream>
#include <vector>
#include <limits>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "meter_reader/global.h"

// The size of inputting images of the detector
std::vector<int> IMAGE_SHAPE = {1920, 1080};
// The size of visualized prediction
std::vector<int> RESULT_SHAPE = {1280, 720};
// The size of inputting images of the segmenter,
// also the size of circular meters
std::vector<int> METER_SHAPE = {512, 512};

// The configuration information of a meter,
// composed of scale value, range, unit
#define METER_TYPE_NUM 2
MeterConfig_T meter_config[METER_TYPE_NUM] = {
{25.0f/50.0f, 25.0f,  "(MPa)"},
{1.6f/32.0f,  1.6f,   "(MPa)"}
};
