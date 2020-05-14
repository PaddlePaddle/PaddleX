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

#include <functional>
#include <iostream>
#include <numeric>

#include "yaml-cpp/yaml.h"

#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

#include <inference_engine.hpp>
#include "include/paddlex/config_parser.h"
#include "include/paddlex/results.h"
#include "include/paddlex/transforms.h"
using namespace InferenceEngine;

namespace PaddleX {

class Model {
 public:
  void Init(const std::string& model_dir,
            const std::string& cfg_dir,
            std::string device) {
    create_predictor(model_dir, cfg_dir,  device);
  }

  void create_predictor(const std::string& model_dir,
                        const std::string& cfg_dir,
                        std::string device);

  bool load_config(const std::string& model_dir);

  bool preprocess(cv::Mat* input_im, ImageBlob* blob);

  bool predict(cv::Mat* im, ClsResult* result);

  std::string type;
  std::string name;
  std::vector<std::string> labels;
  Transforms transforms_;
  Blob::Ptr inputs_;
  Blob::Ptr output_
  CNNNetwork network_;
};
}  // namespce of PaddleX
