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

#include "paddle_inference_api.h"  // NOLINT

#include "include/paddlex/config_parser.h"
#include "include/paddlex/results.h"
#include "include/paddlex/transforms.h"

#ifdef WITH_ENCRYPTION
#include "paddle_model_decrypt.h"
#include "model_code.h"
#endif

namespace PaddleX {

class Model {
 public:
  void Init(const std::string& model_dir,
            bool use_gpu = false,
            bool use_trt = false,
            int gpu_id = 0,
            std::string key = "") {
    create_predictor(model_dir, use_gpu, use_trt, gpu_id, key);
  }

  void create_predictor(const std::string& model_dir,
                        bool use_gpu = false,
                        bool use_trt = false,
                        int gpu_id = 0,
                        std::string key = "");

  bool load_config(const std::string& model_dir);

  bool preprocess(const cv::Mat& input_im, ImageBlob* blob);

  bool predict(const cv::Mat& im, ClsResult* result);

  bool predict(const cv::Mat& im, DetResult* result);

  bool predict(const cv::Mat& im, SegResult* result);

  bool postprocess(SegResult* result);

  bool postprocess(DetResult* result);

  std::string type;
  std::string name;
  std::map<int, std::string> labels;
  Transforms transforms_;
  ImageBlob inputs_;
  std::vector<float> outputs_;
  std::unique_ptr<paddle::PaddlePredictor> predictor_;
};
}  // namespce of PaddleX
