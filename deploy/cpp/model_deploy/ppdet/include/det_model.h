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
#include <string>

#include "yaml-cpp/yaml.h"

#include "model_deploy/common/include/base_model.h"
#include "model_deploy/ppdet/include/det_postprocess.h"
#include "model_deploy/ppdet/include/det_preprocess.h"


namespace PaddleDeploy {
class DetModel : public Model {
 private:
  const std::string model_type;
  bool DetParserTransforms(const YAML::Node &preprocess_op);

 public:
  explicit DetModel(const std::string model_type) : model_type(model_type) {
    std::cerr << "init DetModel,model_type=" << model_type << std::endl;
  }

  bool YamlConfigInit(const std::string &cfg_file);

  bool PreProcessInit();

  bool PostProcessInit(bool use_cpu_nms);
};

}  // namespace PaddleDeploy
