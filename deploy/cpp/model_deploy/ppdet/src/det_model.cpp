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
#include "model_deploy/ppdet/include/det_model.h"

namespace PaddleDeploy {

// regist class (model_type, class_name)
REGISTER_CLASS(det, DetModel);

bool DetModel::DetParserTransforms(const YAML::Node& preprocess_op) {
  if (!preprocess_op["type"].IsDefined()) {
    std::cerr << "preprocess no type" << std::endl;
    return false;
  }

  std::string preprocess_op_type = preprocess_op["type"].as<std::string>();
  if (preprocess_op_type == "Normalize") {
    yaml_config_["transforms"]["Convert"]["dtype"] = "float";
    std::vector<float> mean =
            preprocess_op["mean"].as<std::vector<float>>();
    std::vector<float> std_value =
            preprocess_op["std"].as<std::vector<float>>();
    yaml_config_["transforms"]["Normalize"]["is_scale"] =
        preprocess_op["is_scale"].as<bool>();
    for (int i = 0; i < mean.size(); i++) {
      yaml_config_["transforms"]["Normalize"]["mean"].push_back(mean[i]);
      yaml_config_["transforms"]["Normalize"]["std"].push_back(std_value[i]);
      yaml_config_["transforms"]["Normalize"]["min_val"].push_back(0);
      yaml_config_["transforms"]["Normalize"]["max_val"].push_back(255);
    }
  } else if (preprocess_op_type == "Permute") {
    yaml_config_["transforms"]["Permute"]["is_permute"] = true;
    if (preprocess_op["to_bgr"].as<bool>() == true) {
      yaml_config_["transforms"]["RGB2BGR"]["is_rgb2bgr"] = true;
    }
  } else if (preprocess_op_type == "Resize") {
    int max_size = preprocess_op["max_size"].as<int>();
    if (max_size != 0 && (
        yaml_config_["model_name"].as<std::string>() == "RCNN" ||
        yaml_config_["model_name"].as<std::string>() == "RetinaNet")) {
      yaml_config_["transforms"]["ResizeByShort"]["target_size"] =
          preprocess_op["target_size"].as<int>();
      yaml_config_["transforms"]["ResizeByShort"]["max_size"] = max_size;
      yaml_config_["transforms"]["ResizeByShort"]["interp"] =
          preprocess_op["interp"].as<int>();
      if (preprocess_op["image_shape"].IsDefined()) {
        yaml_config_["transforms"]["Padding"]["width"] = max_size;
        yaml_config_["transforms"]["Padding"]["height"] = max_size;
      }
    } else {
      yaml_config_["transforms"]["Resize"]["width"] =
          preprocess_op["target_size"].as<int>();
      yaml_config_["transforms"]["Resize"]["height"] =
          preprocess_op["target_size"].as<int>();
      yaml_config_["transforms"]["Resize"]["interp"] =
          preprocess_op["interp"].as<int>();
      yaml_config_["transforms"]["Resize"]["max_size"] = max_size;
    }
  } else if (preprocess_op_type == "PadStride") {
    yaml_config_["transforms"]["Padding"]["stride"] =
        preprocess_op["stride"].as<int>();
  } else {
    std::cerr << preprocess_op["type"].as<std::string>()
              << " :Can't parser" << std::endl;
    return false;
  }
  return true;
}

void DetModel::YamlConfigInit(const std::string& cfg_file) {
  YAML::Node det_config = YAML::LoadFile(cfg_file);

  yaml_config_["model_format"] = "Paddle";
  // arch support value:YOLO, SSD, RetinaNet, RCNN, Face
  if (!det_config["arch"].IsDefined()) {
    std::cerr << "Fail to find arch in PaddleDection yaml file" << std::endl;
    return;
  } else if (!det_config["label_list"].IsDefined()) {
    std::cerr << "Fail to find label_list in "
              << "PaddleDection yaml file"
              << std::endl;
    return;
  }
  yaml_config_["model_name"] = det_config["arch"].as<std::string>();
  yaml_config_["toolkit"] = "PaddleDetection";
  yaml_config_["toolkit_version"] = "Unknown";

  int i = 0;
  for (const auto& label : det_config["label_list"]) {
    yaml_config_["labels"][i] = label.as<std::string>();
    i++;
  }

  // Preprocess support Normalize, Permute, Resize, PadStride, Convert
  if (det_config["Preprocess"].IsDefined()) {
    YAML::Node preprocess_info = det_config["Preprocess"];
    for (const auto& preprocess_op : preprocess_info) {
      if (!DetParserTransforms(preprocess_op)) {
        std::cerr << "Fail to parser PaddleDetection "
                  << "transforms of config.yaml"
                  << std::endl;
        return;
      }
    }
  } else {
    std::cerr << "No Preprocess in  PaddleDection yaml file"
              << std::endl;
  }
}

// void DetModel::Init(const std::string &cfg_file){

// }

void DetModel::PreProcessInit() {
  preprocess_ = std::make_shared<DetPreProcess>();
  preprocess_->Init(yaml_config_);
}

void DetModel::PostProcessInit(bool use_cpu_nms) {
  postprocess_ = std::make_shared<DetPostProcess>();
  postprocess_->Init(yaml_config_, use_cpu_nms);
}

}  // namespace PaddleDeploy
