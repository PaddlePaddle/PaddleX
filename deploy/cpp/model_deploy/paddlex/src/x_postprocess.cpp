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

#include "model_deploy/paddlex/include/x_postprocess.h"

namespace PaddleDeploy {

bool XPostProcess::Init(const YAML::Node& yaml_config) {
  model_type_ = yaml_config["model_type"].as<std::string>();
  if (model_type_ == "segmenter") {
    seg_post_process.Init(yaml_config);
  } else if (model_type_ == "detector") {
    det_post_process.Init(yaml_config);
  } else if (model_type_ == "classifier") {
    clas_post_process.Init(yaml_config);
  } else {
    std::cerr << "[ERROR] Unexpected model type '"
              << model_type_ << "' for PaddleX"
              << std::endl;
    return false;
  }
  return true;
}

bool XPostProcess::Run(const std::vector<DataBlob>& outputs,
                         const std::vector<ShapeInfo>& shape_infos,
                         std::vector<Result>* results, int thread_num) {
  results->clear();
  results->resize(shape_infos.size());
  if (model_type_ == "segmenter") {
    return seg_post_process.Run(outputs, shape_infos, results, thread_num);
  } else if (model_type_ == "classifier") {
    return clas_post_process.Run(outputs, shape_infos, results, thread_num);
  } else if (model_type_ == "detector") {
    return det_post_process.Run(outputs, shape_infos, results, thread_num);
  } else {
    std::cerr << "[ERROR] Unexpected model_type: '"
              << model_type_ << "' in postprocess"
              << std::endl;
    return false;
  }
  return true;
}
}  //  namespace PaddleDeploy
