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

#include <arm_neon.h>
#include <paddle_api.h>

#include <functional>
#include <iostream>
#include <numeric>
#include <map>
#include <string>
#include <memory>

#include "include/paddlex/config_parser.h"
#include "include/paddlex/results.h"
#include "include/paddlex/transforms.h"



#include "yaml-cpp/yaml.h"




#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif




namespace PaddleX {

class Model {
 public:
  void Init(const std::string& model_dir,
            const std::string& cfg_dir,
            int thread_num) {
    create_predictor(model_dir, cfg_dir, thread_num);
  }

  void create_predictor(const std::string& model_dir,
                        const std::string& cfg_dir,
                        int thread_num);

  bool load_config(const std::string& model_dir);

  bool preprocess(cv::Mat* input_im, ImageBlob* inputs);

  bool predict(const cv::Mat& im, ClsResult* result);

  bool predict(const cv::Mat& im, DetResult* result);

  bool predict(const cv::Mat& im, SegResult* result);


  std::string type;
  std::string name;
  std::map<int, std::string> labels;
  Transforms transforms_;
  ImageBlob inputs_;
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
};
}  // namespace PaddleX
