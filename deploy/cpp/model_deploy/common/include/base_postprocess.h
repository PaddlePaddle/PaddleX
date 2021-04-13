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

#include <vector>

#include "model_deploy/common/include/output_struct.h"
#include "yaml-cpp/yaml.h"

namespace PaddleDeploy {

class BasePostProcess {
 public:
  bool use_cpu_nms_;

  virtual void Init(const YAML::Node &yaml_config,
                    bool use_cpu_nms = false) {
    use_cpu_nms_ = use_cpu_nms;
  }

  virtual bool Run(const std::vector<DataBlob>& outputs,
                   const std::vector<ShapeInfo>& shape_infos,
                   std::vector<Result>* results, int thread_num = 1) = 0;
};

}  // namespace PaddleDeploy
