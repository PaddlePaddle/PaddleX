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

#pragma once

#include <vector>

// The configuration information of a meter, composed of scale value,
// range, unit.
typedef struct MeterConfig {
  float scale_value;
  float range;
  char  str[10];
} MeterConfig_T;

// The size of inputting images of the detector
extern std::vector<int> IMAGE_SHAPE;
// The size of visualized prediction
extern std::vector<int> RESULT_SHAPE;
// The size of inputting images of the segmenter,
// also the size of circular meters.
extern std::vector<int> METER_SHAPE;
extern MeterConfig_T meter_config[];
// The type of a meter is estimated by a threshold. If the number of scales
// in a meter is greater than or equal to the threshold, the meter is
// belong to the former type. Otherwize, the latter.
#define TYPE_THRESHOLD 40
