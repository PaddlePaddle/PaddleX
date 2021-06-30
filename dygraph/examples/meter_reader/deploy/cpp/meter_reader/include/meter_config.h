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
#include <string>
#include <map>

struct MeterConfig {
  float scale_interval_value_;
  float range_;
  std::string unit_;

  MeterConfig() {}

  MeterConfig(const float &scale_interval_value,
              const float &range,
              const std::string &unit) :
    scale_interval_value_(scale_interval_value),
    range_(range), unit_(unit) {}
};

struct MeterResult {
  // the number of scales
  int num_scales_;
  // the pointer location relative to the scales
  float pointed_scale_;

  MeterResult() {}

  MeterResult(const int &num_scales, const float &pointed_scale) :
    num_scales_(num_scales), pointed_scale_(pointed_scale) {}
};

// The size of inputting images of the segmenter.
extern std::vector<int> METER_SHAPE;  // height x width
// Center of a circular meter
extern std::vector<int> CIRCLE_CENTER;  // height x width
// Radius of a circular meter
extern int CIRCLE_RADIUS;
extern float PI;

// During the postprocess phase, annulus formed by the radius from
// 130 to 250 of a circular meter will be converted to a rectangle.
// So the height of the rectangle is 120.
extern int RECTANGLE_HEIGHT;
// The width of the rectangle is 1570, that is to say the perimeter
// of a circular meter.
extern int RECTANGLE_WIDTH;

// The configuration information of a meter,
// composed of scale value, range, unit
extern int TYPE_THRESHOLD;
extern std::vector<MeterConfig> METER_CONFIG;
extern std::map<std::string, uint8_t> SEG_CNAME2CLSID;
