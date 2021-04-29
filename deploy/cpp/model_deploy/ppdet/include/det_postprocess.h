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

#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "yaml-cpp/yaml.h"

#include "model_deploy/common/include/base_postprocess.h"
#include "model_deploy/common/include/output_struct.h"

namespace PaddleDeploy {

class DetPostProcess : public BasePostProcess {
 public:
  bool Init(const YAML::Node& yaml_config);

  bool ProcessBbox(const std::vector<DataBlob>& outputs,
                   const std::vector<ShapeInfo>& shape_infos,
                   std::vector<Result>* results, int thread_num = 1);

  virtual bool Run(const std::vector<DataBlob>& outputs,
                   const std::vector<ShapeInfo>& shape_infos,
                   std::vector<Result>* results, int thread_num = 1);

 private:
  std::string model_arch_;
  std::vector<std::string> labels_;
};

}  // namespace PaddleDeploy
