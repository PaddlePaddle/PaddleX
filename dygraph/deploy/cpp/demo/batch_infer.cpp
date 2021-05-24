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

#include <fstream>
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
  PaddleDeploy::PaddleEngineConfig engine_config;
  engine_config.model_filename = "resnet50/inference.pdmodel";
  engine_config.params_filename = "resnet50/inference.pdiparams";
  engine_config.use_gpu = true;
  engine_config.max_batch_size = 8;
  model->PaddleEngineInit(engine_config);

  // prepare data
  std::string image_list = "resnet50/file_list.txt";
  std::vector<cv::Mat> imgs;
  if (image_list != "") {
    std::ifstream inf(image_list);
    if (!inf) {
      std::cerr << "Fail to open file " << image_list << std::endl;
      return -1;
    }
    std::string image_path;
    while (getline(inf, image_path)) {
      imgs.push_back(std::move(cv::imread(image_path)));
    }
  }

  // batch predict
  std::vector<PaddleDeploy::Result> results;
  int batch_size = 8;
  for (int i = 0; i < imags.size(); i += batch_size) {
    int im_vec_size = std::min(static_cast<int>(imgs.size()), i + batch_size);
    std::vector<cv::Mat> im_vec(imgs.begin() + i,
                                imgs.begin() + im_vec_size);

    model->Predict(im_vec, &results, 8);

    // print result
    for (auto j = 0; j < results.size(); ++j) {
      std::cout << "Result for sample " << j << std::endl;
      std::cout << results[j] << std::endl;
    }
    results.clear();
  }

  return 0;
}
