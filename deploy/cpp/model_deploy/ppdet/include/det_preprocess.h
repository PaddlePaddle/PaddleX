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

#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "model_deploy/common/include/base_preprocess.h"
#include "model_deploy/common/include/output_struct.h"

namespace PaddleDeploy {
class DetPreProcess : public BasePreProcess {
 public:
  bool Init(const YAML::Node &yaml_config);

  virtual bool PrepareInputs(const std::vector<ShapeInfo>& shape_infos,
                             std::vector<cv::Mat>* imgs,
                             std::vector<DataBlob>* inputs,
                             int thread_num = 1);
  bool PrepareInputsForV2(const std::vector<cv::Mat>& imgs,
                            const std::vector<ShapeInfo>& shape_infos,
                            std::vector<DataBlob>* inputs,
                            int thread_num = 1);
  bool PrepareInputsForRCNN(const std::vector<cv::Mat>& imgs,
                            const std::vector<ShapeInfo>& shape_infos,
                            std::vector<DataBlob>* inputs,
                            int thread_num = 1);
  bool PrepareInputsForYOLO(const std::vector<cv::Mat>& imgs,
                            const std::vector<ShapeInfo>& shape_infos,
                            std::vector<DataBlob>* inputs,
                            int thread_num = 1);
  virtual bool Run(std::vector<cv::Mat>* imgs,
                   std::vector<DataBlob>* inputs,
                   std::vector<ShapeInfo>* shape_info,
                   int thread_num = 1);

 private:
  std::string model_arch_;
  std::string version_;
};
}  // namespace PaddleDeploy
