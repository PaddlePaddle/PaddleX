//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <iostream>

#include "yaml-cpp/yaml.h"

namespace PaddleDeploy {

void SegNormalize(const YAML::Node& src, YAML::Node* dst) {
  std::vector<float> mean({0.5, 0.5, 0.5});
  std::vector<float> std({0.5, 0.5, 0.5});
  if (src["mean"].IsDefined()) {
    mean = src["mean"].as<std::vector<float>>();
  }
  if (src["std"].IsDefined()) {
    std = src["std"].as<std::vector<float>>();
  }
  for (auto i = 0; i < mean.size(); ++i) {
    (*dst)["transforms"]["Normalize"]["mean"].push_back(mean[i]);
    (*dst)["transforms"]["Normalize"]["std"].push_back(std[i]);
  }
}

void SegResize(const YAML::Node& src, YAML::Node* dst) {
  std::vector<int> target_size{512, 512};
  int interp = 1;
  if (src["target_size"].IsDefined()) {
    target_size = src["target_size"].as<std::vector<int>>();
  }
  if (src["interp"].IsDefined()) {
    interp = src["interp"].as<int>();
  }
  (*dst)["transforms"]["Resize"]["width"] = target_size[0];
  (*dst)["transforms"]["Resize"]["height"] = target_size[1];
  (*dst)["transforms"]["Resize"]["interp"] = interp;
}

void SegPadding(const YAML::Node& src, YAML::Node* dst) {
  assert(src["target_size"].IsDefined());
  assert(src["target_size"].IsScalar() || src["target_size"].IsSequence());
  std::vector<int> target_size = src["target_size"].as<std::vector<int>>();
  std::vector<float> im_padding_value({127.5, 127.5, 127.5});
  if (src["im_padding_value"].IsDefined()) {
    im_padding_value = src["im_padding_value"].as<std::vector<float>>();
  }
  int w = 0;
  int h = 0;
  if (src["target_size"].IsScalar()) {
    w = src["target_size"].as<int>();
    h = src["target_size"].as<int>();
  } else if (src["target_size"].IsSequence()) {
    w = src["target_size"].as<std::vector<int>>()[0];
    h = src["target_size"].as<std::vector<int>>()[1];
  }
  (*dst)["transforms"]["Padding"]["width"] = w;
  (*dst)["transforms"]["Padding"]["height"] = h;
  (*dst)["transforms"]["Padding"]["im_padding_value"] = im_padding_value;
}
}  // namespace PaddleDeploy
