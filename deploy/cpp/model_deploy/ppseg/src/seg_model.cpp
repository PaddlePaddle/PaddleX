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
#include "model_deploy/ppseg/include/seg_model.h"
#include "model_deploy/ppseg/include/seg_standard_config.h"

namespace PaddleDeploy {

bool SegModel::GenerateTransformsConfig(const YAML::Node& src) {
  yaml_config_["transforms"]["BGR2RGB"] = YAML::Null;
  yaml_config_["transforms"]["Convert"]["dtype"] = "float";
  for (const auto& op : src) {
    assert(op["type"].IsDefined());
    std::string op_name = op["type"].as<std::string>();
    if (op_name == "Normalize") {
      SegNormalize(op, &yaml_config_);
    } else if (op_name == "Resize") {
      SegResize(op, &yaml_config_);
    } else if (op_name == "Padding") {
      SegPadding(op, &yaml_config_);
    } else {
      std::cerr << "Unexpected transforms op name: '"
                << op_name << "'" << std::endl;
      return false;
    }
  }
  yaml_config_["transforms"]["Permute"] = YAML::Null;
  return true;
}

bool SegModel::YamlConfigInit(const std::string& cfg_file) {
  YAML::Node seg_config = YAML::LoadFile(cfg_file);

  yaml_config_["model_format"] = "Paddle";
  yaml_config_["toolkit"] = "PaddleSeg";
  yaml_config_["toolkit_version"] = "Unknown";

  // Generate Standard Transforms Configuration
  if (!GenerateTransformsConfig(seg_config["Deploy"]["transforms"])) {
    std::cerr << "Fail to generate standard configuration "
              << "of tranforms" << std::endl;
    return false;
  }
  return true;
}

bool SegModel::PreprocessInit() {
  preprocess_ = std::make_shared<SegPreprocess>();
  if (!preprocess_->Init(yaml_config_))
    return false;
  return true;
}

bool SegModel::PostprocessInit() {
  postprocess_ = std::make_shared<SegPostprocess>();
  if (!postprocess_->Init(yaml_config_))
    return false;
  return true;
}

}  // namespace PaddleDeploy
