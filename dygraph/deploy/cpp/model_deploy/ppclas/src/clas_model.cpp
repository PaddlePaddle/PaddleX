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
#include "model_deploy/ppclas/include/clas_model.h"

namespace PaddleDeploy {

bool ClasModel::YamlConfigInit(const std::string& cfg_file,
                               const std::string key) {
  if ("" == key) {
    yaml_config_ = YAML::LoadFile(cfg_file);
  } else {
#ifdef PADDLEX_DEPLOY_ENCRYPTION
    std::string cfg = decrypt_file(cfg_file.c_str(), key.c_str());
    yaml_config_ = YAML::Load(cfg);
#else
     std::cerr << "Don't open encryption on compile" << std::endl;
    return false;
#endif  // PADDLEX_DEPLOY_ENCRYPTION
  }
  return true;
}

bool ClasModel::PreprocessInit() {
  preprocess_ = std::make_shared<ClasPreprocess>();
  if (!preprocess_->Init(yaml_config_)) {
    return false;
  }
  return true;
}

bool ClasModel::PostprocessInit() {
  postprocess_ = std::make_shared<ClasPostprocess>();
  if (!postprocess_->Init(yaml_config_))
    return false;
  return true;
}

}  // namespace PaddleDeploy
