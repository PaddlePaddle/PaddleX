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

#include "model_deploy/paddlex/include/x_preprocess.h"

namespace PaddleDeploy {

bool XPreprocess::Init(const YAML::Node& yaml_config) {
  model_type_ = yaml_config["model_type"].as<std::string>();
  model_name_ = yaml_config["model_name"].as<std::string>();

  if (model_type_ == "segmenter") {
    return seg_preprocess.Init(yaml_config);
  } else if (model_type_ == "classifier") {
    return clas_preprocess.Init(yaml_config);
  } else if (model_type_ == "detector") {
    return det_preprocess.Init(yaml_config);
  } else {
    std::cerr << "[ERROR] Unexpected model_type: '"
              << model_type_ << "' in preprocess Init"
              << std::endl;
    return false;
  }
  return true;
}

bool XPreprocess::Run(std::vector<cv::Mat>* imgs,
                        std::vector<DataBlob>* inputs,
                        std::vector<ShapeInfo>* shape_infos, int thread_num) {
  if (model_type_ == "segmenter") {
    return seg_preprocess.Run(imgs, inputs, shape_infos, thread_num);
  } else if (model_type_ == "classifier") {
    return clas_preprocess.Run(imgs, inputs, shape_infos, thread_num);
  } else if (model_type_ == "detector") {
    return det_preprocess.Run(imgs, inputs, shape_infos, thread_num);
  } else {
    std::cerr << "[ERROR] Unexpected model_type: '"
              << model_type_ << "' in preprocess"
              << std::endl;
    return false;
  }
  return true;
}

}  //  namespace PaddleDeploy
