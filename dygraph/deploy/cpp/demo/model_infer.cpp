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

#include <glog/logging.h>
#include <omp.h>
#include <memory>
#include <string>
#include <fstream>

#include "model_deploy/common/include/paddle_deploy.h"

DEFINE_string(model_filename, "", "Path of det inference model");
DEFINE_string(params_filename, "", "Path of det inference params");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(model_type, "", "model type");
DEFINE_string(image, "", "Path of test image file");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_bool(use_trt, false, "Infering with TensorRT");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << "model_type=" << FLAGS_model_type << std::endl;

  // create model
  std::shared_ptr<PaddleDeploy::Model> model =
        PaddleDeploy::CreateModel(FLAGS_model_type);

  std::cout << "start model init " << std::endl;
  // model init
  model->Init(FLAGS_cfg_file);

  std::cout << "start engine init " << std::endl;
  // inference engine init
  PaddleDeploy::PaddleEngineConfig engine_config;
  engine_config.model_filename = FLAGS_model_filename;
  engine_config.params_filename = FLAGS_params_filename;
  engine_config.use_gpu = FLAGS_use_gpu;
  engine_config.gpu_id = FLAGS_gpu_id;
  engine_config.use_trt = FLAGS_use_trt;
  if (FLAGS_use_trt) {
    engine_config.precision = 0;
  }
  model->PaddleEngineInit(engine_config);

  // prepare data
  std::vector<cv::Mat> imgs;
  imgs.push_back(std::move(cv::imread(FLAGS_image)));

  // predict
  std::vector<PaddleDeploy::Result> results;
  model->Predict(imgs, &results, 1);

  for (auto j = 0; j < results.size(); ++j) {
    std::cout << "Result for sample " << j << std::endl;
    std::cout << results[j] << std::endl;
  }
  return 0;
}
