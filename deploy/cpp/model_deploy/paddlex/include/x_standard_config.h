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

void XEssential(const YAML::Node& src, YAML::Node* dst) {
  assert(src["Transforms"].IsDefined());
  if (src["TransformsMode"].IsDefined()) {
    std::string mode = src["TransformsMode"].as<std::string>();
    if (mode == "RGB") {
      (*dst)["transforms"]["BGR2RGB"] = YAML::Null;
    } else if (mode != "BGR") {
      std::cerr << "[Error] Only support RGB or BGR of "
                << "TransformsMode" << std::endl;
    }
  }
  (*dst)["transforms"]["Convert"]["dtype"] = "float";
}


void XNormalize(const YAML::Node& src, YAML::Node* dst) {
  // check data format
  assert(src["mean"].IsDefined());
  assert(src["std"].IsDefined());

  // normalize
  std::vector<float> mean = src["mean"].as<std::vector<float>>();
  std::vector<float> std = src["std"].as<std::vector<float>>();
  for (auto i = 0; i < mean.size(); ++i) {
    (*dst)["transforms"]["Normalize"]["mean"].push_back(mean[i]);
    (*dst)["transforms"]["Normalize"]["std"].push_back(std[i]);
  }
}

void XResize(const YAML::Node& src, YAML::Node* dst) {
  // check data format
  assert(src["target_size"].IsDefined());
  int w = 0;
  int h = 0;
  if (src["target_size"].IsScalar()) {
    w = src["target_size"].as<int>();
    h = src["target_size"].as<int>();
  } else if (src["target_size"].IsSequence()) {
    w = src["target_size"].as<std::vector<int>>()[0];
    h = src["target_size"].as<std::vector<int>>()[1];
  } else {
    std::cerr << "[ERROR] Unexpected value type of `target_size`" << std::endl;
    assert(false);
  }
  int interp = 1;
  if (src["interp"].IsDefined()) {
    std::string interp_str = src["interp"].as<std::string>();
    if (interp_str == "NEAREST") {
      interp = 0;
    } else if (interp_str == "LINEAR") {
      interp = 1;
    } else if (interp_str == "CUBIC") {
      interp = 2;
    } else if (interp_str == "AREA") {
      interp = 3;
    } else if (interp_str == "LANCZOS4") {
      interp = 4;
    } else {
      std::cerr << "[ERROR] Unexpected interpolation method: '"
                << interp_str << "'" << std::endl;
      assert(false);
    }
  }
  (*dst)["transforms"]["Resize"]["width"] = w;
  (*dst)["transforms"]["Resize"]["height"] = h;
  (*dst)["transforms"]["Resize"]["interp"] = interp;
  (*dst)["transforms"]["Resize"]["use_scale"] = false;
}

void XResizeByLong(const YAML::Node& src, YAML::Node* dst) {
  // check data format
  assert(src["long_size"].IsDefined());

  int long_size = src["long_size"].as<int>();
  (*dst)["transforms"]["ResizeByLong"]["target_size"] = long_size;
  (*dst)["transforms"]["ResizeByLong"]["interp"] = 1;
}

void XResizeByShort(const YAML::Node& src, YAML::Node* dst) {
  // check data format
  assert(src["max_size"].IsDefined());
  assert(src["short_size"].IsDefined());

  int max_size = src["max_size"].as<int>();
  int target_size = src["short_size"].as<int>();
  int interp = 1;
  (*dst)["transforms"]["ResizeByShort"]["max_size"] = max_size;
  (*dst)["transforms"]["ResizeByShort"]["target_size"] = target_size;
  (*dst)["transforms"]["ResizeByShort"]["interp"] = interp;
  (*dst)["transforms"]["ResizeByShort"]["use_scale"] = false;
}

void XPadding(const YAML::Node& src, YAML::Node* dst) {
  if (src["coarsest_stride"].IsDefined()) {
    (*dst)["transforms"]["Padding"]["stride"] =
                        src["coarsest_stride"].as<int>();
  } else if (src["target_size"].IsDefined()) {
    assert(src["target_size"].IsScalar() || src["target_size"].IsSequence());
    if (src["target_size"].IsScalar()) {
      (*dst)["transforms"]["Padding"]["width"] = src["target_size"].as<int>();
      (*dst)["transforms"]["Padding"]["height"] = src["target_size"].as<int>();
    } else {
      std::vector<int> target_size = src["target_size"].as<std::vector<int>>();
      (*dst)["transforms"]["Padding"]["width"] = target_size[0];
      (*dst)["transforms"]["Padding"]["height"] = target_size[1];
    }
  } else {
    std::cerr << "[Error] As least one of coarsest_stride/"
              << "target_size must be defined for Padding"
              << std::endl;
    assert(false);
  }

  if (src["im_padding_value"].IsDefined()) {
    (*dst)["transforms"]["Padding"]["im_padding_value"] =
            src["im_padding_value"].as<std::vector<float>>();
  }
}

void XCenterCrop(const YAML::Node& src, YAML::Node* dst) {
  assert(src["crop_size"].IsDefined());
  assert(src["crop_size"].IsScalar() || src["crop_size"].IsSequence());
  if (src["crop_size"].IsScalar()) {
    (*dst)["transforms"]["CenterCrop"]["width"] = src["crop_size"].as<int>();
    (*dst)["transforms"]["CenterCrop"]["height"] = src["crop_size"].as<int>();
  }
}

}  // namespace PaddleDeploy
