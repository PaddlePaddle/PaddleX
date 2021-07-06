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


#include "meter_reader/include/meter_config.h"

std::vector<int> METER_SHAPE = {512, 512};  // height x width
std::vector<int> CIRCLE_CENTER = {256, 256};
int CIRCLE_RADIUS = 250;
float PI = 3.1415926536;
int RECTANGLE_HEIGHT = 120;
int RECTANGLE_WIDTH = 1570;

int TYPE_THRESHOLD = 40;
std::vector<MeterConfig> METER_CONFIG = {
  MeterConfig(25.0f/50.0f, 25.0f, "(MPa)"),
  MeterConfig(1.6f/32.0f,  1.6f,   "(MPa)")
};

std::map<std::string, uint8_t> SEG_CNAME2CLSID = {
  {"background", 0}, {"pointer", 1}, {"scale", 2}
};
