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
#include <memory>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "model_deploy/common/include/output_struct.h"
#include "model_deploy/common/include/transforms.h"

namespace PaddleDeploy {

class BasePreprocess {
 public:
  virtual bool Init(const YAML::Node& yaml_config) {
    if (!BuildTransform(yaml_config))
      return false;
    return true;
  }

  bool PreprocessImages(const std::vector<ShapeInfo>& shape_infos,
                        std::vector<cv::Mat>* imgs,
                        int thread_num = 1);

  bool ShapeInfer(const std::vector<cv::Mat>& imgs,
                  std::vector<ShapeInfo>* shape_infos,
                  int thread_num = 1);

  virtual bool Run(std::vector<cv::Mat>* imgs,
                   std::vector<DataBlob>* inputs,
                   std::vector<ShapeInfo>* shape_info,
                   int thread_num = 1) = 0;

 protected:
  bool BuildTransform(const YAML::Node& yaml_config);
  std::vector<std::shared_ptr<Transform>> transforms_;

 private:
  std::shared_ptr<Transform> CreateTransform(const std::string& name);
  Padding batch_padding_;
  Permute permute_;
};

}  // namespace PaddleDeploy
