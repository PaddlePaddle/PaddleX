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

#include <string>
#include <vector>

#include "model_deploy/common/include/paddle_deploy.h"

int main(int argc, char** argv) {
  // create model
  std::string model_type = "det";
  std::shared_ptr<PaddleDeploy::Model> model =
          PaddleDeploy::CreateModel(model_type);

  // model init
  std::string cfg_file = "resnet50/deploy.yml";
  model->Init(cfg_file);

  // inference engine init
  PaddleEngineConfig engine_config;
  engine_config.model_filename = "resnet50/inference.pdmodel";
  engine_config.params_filename = "resnet50/inference.pdiparams";
  engine_config.use_gpu = true;
  engine_config.use_trt = true;
  engine_config.precision = 0;
  model->PaddleEngineInit(engine_config);

  // prepare data
  std::string image_path = "resnet50/test.jpg"
  std::vector<cv::Mat> imgs;
  imgs.push_back(std::move(cv::imread(image_paths[j])));

  // predict
  std::vector<PaddleDeploy::Result> results;
  model->Predict(imgs, &results, 1);

  for (auto j = 0; j < results.size(); ++j) {
    std::cout << "Result for sample " << j << std::endl;
    std::cout << results[j] << std::endl;
  }

  return 0;
}
