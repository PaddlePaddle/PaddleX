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

#include <gflags/gflags.h>
#include <string>
#include <vector>

#include "model_deploy/common/include/paddle_deploy.h"

DEFINE_string(model_filename, "", "Path of det inference model");
DEFINE_string(params_filename, "", "Path of det inference params");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(model_type, "", "model type");
DEFINE_string(image, "", "Path of test image file");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_bool(use_trt, false, "Infering with TensorRT");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);

  // create model
  std::shared_ptr<PaddleDeploy::Model> model =
        PaddleDeploy::CreateModel(FLAGS_model_type);

  // model init
  model->Init(FLAGS_cfg_file);

  // inference engine init
  PaddleDeploy::PaddleEngineConfig engine_config;
  engine_config.model_filename = FLAGS_model_filename;
  engine_config.params_filename = FLAGS_params_filename;
  engine_config.gpu_id = FLAGS_gpu_id;
  engine_config.use_trt = FLAGS_use_trt;
  if (FLAGS_use_trt) {
    /*Set TensorRT data precision
      0: FP32
      1: FP16
      2: Int8
    */
    engine_config.precision = 0;
    engine_config.min_subgraph_size = 10;
    engine_config.max_workspace_size = 1 << 30;
    if ("clas" == FLAGS_model_type) {
      // Adjust shape according to the actual model
      engine_config.min_input_shape["inputs"] = {1, 3, 224, 224};
      engine_config.max_input_shape["inputs"] = {1, 3, 224, 224};;
      engine_config.optim_input_shape["inputs"] = {1, 3, 224, 224};;
    } else if ("det" == FLAGS_model_type) {
      // Adjust shape according to the actual model
      engine_config.min_input_shape["image"] = {1, 3, 608, 608};
      engine_config.max_input_shape["image"] = {1, 3, 608, 608};
      engine_config.optim_input_shape["image"] = {1, 3, 608, 608};
    } else if ("seg" == FLAGS_model_type) {
      engine_config.min_input_shape["x"] = {1, 3, 100, 100};
      engine_config.max_input_shape["x"] = {1, 3, 2000, 2000};
      engine_config.optim_input_shape["x"] = {1, 3, 1024, 1024};
      // Additional nodes need to be added, pay attention to the output prompt
    }
  }
  model->PaddleEngineInit(engine_config);

  // prepare data
  std::vector<cv::Mat> imgs;
  imgs.push_back(std::move(cv::imread(FLAGS_image)));

  // predict
  std::vector<PaddleDeploy::Result> results;
  model->Predict(imgs, &results, 1);

  std::cout << results[0] << std::endl;

  return 0;
}
