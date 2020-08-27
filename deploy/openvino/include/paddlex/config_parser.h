//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

namespace PaddleX {

// Inference model configuration parser
class ConfigPaser {
 public:
  ConfigPaser() {}

  ~ConfigPaser() {}

  bool load_config(const std::string& model_dir,
                   const std::string& cfg = "model.yml") {
    // Load as a YAML::Node
    YAML::Node config;
    config = YAML::LoadFile(model_dir + OS_PATH_SEP + cfg);

    if (config["Transforms"].IsDefined()) {
      YAML::Node transforms_ = config["Transforms"];
    } else {
      std::cerr << "There's no field 'Transforms' in model.yml" << std::endl;
      return false;
    }
    return true;
  }

  YAML::Node Transforms_;
};

}  // namespace PaddleX
