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
#include "model_deploy/paddlex/include/x_model.h"
#include "model_deploy/paddlex/include/x_standard_config.h"

#include <fstream>

namespace PaddleDeploy {

bool PaddleXModel::GenerateTransformsConfig(const YAML::Node& src) {
  XEssential(src, &yaml_config_);
  for (const auto& op : src["Transforms"]) {
    std::string op_name = op.begin()->first.as<std::string>();
    if (op_name == "Normalize") {
      if (src["version"].as<std::string>() >= "2.0.0") {
        yaml_config_["transforms"]["Convert"]["dtype"] = "float";
      }
      XNormalize(op.begin()->second, &yaml_config_);
    } else if (op_name == "ResizeByShort") {
      XResizeByShort(op.begin()->second, &yaml_config_);
    } else if (op_name == "ResizeByLong") {
      XResizeByLong(op.begin()->second, &yaml_config_);
    } else if (op_name == "Padding") {
      if (src["version"].as<std::string>() >= "2.0.0") {
        XPaddingV2(op.begin()->second, &yaml_config_);
      } else {
        XPadding(op.begin()->second, &yaml_config_);
      }
    } else if (op_name == "CenterCrop") {
      XCenterCrop(op.begin()->second, &yaml_config_);
    } else if (op_name == "Resize") {
      XResize(op.begin()->second, &yaml_config_);
    } else {
      std::cerr << "Unexpected transforms op name: '"
                << op_name << "'" << std::endl;
      return false;
    }
  }
  yaml_config_["transforms"]["Permute"] = YAML::Null;
  return true;
}

bool PaddleXModel::YamlConfigInit(const std::string& cfg_file,
                                  const std::string key) {
  if ("" == key) {
    YAML::Node x_config = YAML::LoadFile(cfg_file);
  } else {
    std::string cfg = decrypt_file(cfg_file.c_str(), key.c_str());
    YAML::Node x_config = YAML::Load(cfg);
  }

  yaml_config_["model_format"] = "Paddle";
  yaml_config_["toolkit"] = "PaddleX";
  yaml_config_["version"] = x_config["version"].as<std::string>();
  yaml_config_["model_type"] =
        x_config["_Attributes"]["model_type"].as<std::string>();
  yaml_config_["model_name"] = x_config["Model"].as<std::string>();

  int i = 0;
  for (const auto& label : x_config["_Attributes"]["labels"]) {
    yaml_config_["labels"][i] = label.as<std::string>();
    i++;
  }

  // Generate Standard Transforms Configuration
  if (!GenerateTransformsConfig(x_config)) {
    std::cerr << "Fail to generate standard configuration "
              << "of tranforms" << std::endl;
    return false;
  }
  return true;
}

bool PaddleXModel::PreprocessInit() {
  preprocess_ = std::make_shared<XPreprocess>();
  if (!preprocess_->Init(yaml_config_)) {
    return false;
  }
  return true;
}

bool PaddleXModel::PostprocessInit() {
  postprocess_ = std::make_shared<XPostprocess>();
  if (!postprocess_->Init(yaml_config_)) {
    return false;
  }
  return true;
}

}  // namespace PaddleDeploy
