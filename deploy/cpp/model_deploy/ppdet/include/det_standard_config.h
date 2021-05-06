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

void DetNormalize(const YAML::Node& src, YAML::Node* dst) {
  // check data format
  assert(src["is_scale"].IsDefined());
  assert(src["mean"].IsDefined());
  assert(src["std"].IsDefined());

  // convert data to float
  (*dst)["transforms"]["Convert"]["dtype"] = "float";

  // normalize
  bool is_scale = src["is_scale"].as<bool>();
  std::vector<float> mean = src["mean"].as<std::vector<float>>();
  std::vector<float> std = src["std"].as<std::vector<float>>();
  (*dst)["transforms"]["Normalize"]["is_scale"] = is_scale;
  for (auto i = 0; i < mean.size(); ++i) {
    (*dst)["transforms"]["Normalize"]["mean"].push_back(mean[i]);
    (*dst)["transforms"]["Normalize"]["std"].push_back(std[i]);
  }
}

void DetPermute(const YAML::Node& src, YAML::Node* dst) {
  // check data format
  (*dst)["transforms"]["Permute"] = YAML::Null;;
  if (src["to_bgr"].IsDefined()) {
    if (src["to_bgr"].as<bool>()) {
      (*dst)["transforms"]["RGB2BGR"] = YAML::Null;;
    }
  }
}

// Resize OP for PaddleDetection release/2.0
void DetResize2(const YAML::Node& src,
               YAML::Node* dst) {
  // check data format
  assert(src["target_size"].IsDefined());
  assert(src["interp"].IsDefined());
  assert(src["target_size"].IsSequence());

  bool keep_ratio = src["keep_ratio"].as<bool>();
  std::vector<int> target_size = src["target_size"].as<std::vector<int>>();
  int interp = src["interp"].as<int>();
  assert(interp >= 0 && interp < 5);

  if (keep_ratio) {
    (*dst)["transforms"]["ResizeByShort"]["max_size"] = target_size[1];
    (*dst)["transforms"]["ResizeByShort"]["target_size"] = target_size[0];
    (*dst)["transforms"]["ResizeByShort"]["interp"] = interp;
    if (src["image_shape"].IsDefined()) {
      std::vector<int> image_shape = src["image_shape"].as<std::vector<int>>();
      (*dst)["transforms"]["Padding"]["width"] = image_shape[2];
      (*dst)["transforms"]["Padding"]["height"] = image_shape[1];
    }
  } else {
    (*dst)["transforms"]["Resize"]["width"] = target_size[0];
    (*dst)["transforms"]["Resize"]["height"] = target_size[0];
    (*dst)["transforms"]["Resize"]["interp"] = interp;
    (*dst)["transforms"]["Resize"]["use_scale"] = false;
  }
}

void DetResize(const YAML::Node& src,
               YAML::Node* dst,
               const std::string& model_arch) {
  if (src["keep_ratio"].IsDefined()) {
    DetResize2(src, dst);
    return;
  }
  // check data format
  assert(src["max_size"].IsDefined());
  assert(src["target_size"].IsDefined());
  assert(src["interp"].IsDefined());
  assert(model_arch == "RCNN" || model_arch == "YOLO");

  int max_size = src["max_size"].as<int>();
  int target_size = src["target_size"].as<int>();
  int interp = src["interp"].as<int>();
  assert(interp >= 0 && interp < 5 && target_size > 0);

  if (max_size != 0 && model_arch == "RCNN") {
    (*dst)["transforms"]["ResizeByShort"]["max_size"] = max_size;
    (*dst)["transforms"]["ResizeByShort"]["target_size"] = target_size;
    (*dst)["transforms"]["ResizeByShort"]["interp"] = interp;
  } else {
    (*dst)["transforms"]["Resize"]["width"] = target_size;
    (*dst)["transforms"]["Resize"]["height"] = target_size;
    (*dst)["transforms"]["Resize"]["interp"] = interp;
    (*dst)["transforms"]["Resize"]["use_scale"] = false;
  }
}

void DetPadStride(const YAML::Node& src, YAML::Node* dst) {
  // check data format
  assert(src["stride"].IsDefined());
  if (src["stride"].as<int>() < 0) {
    return;
  }
  (*dst)["transforms"]["Padding"]["stride"] = src["stride"].as<int>();
}

}  // namespace PaddleDeploy
