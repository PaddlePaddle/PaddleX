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
#include <omp.h>
#include <memory>
#include <string>
#include <fstream>

#include "model_deploy/common/include/paddle_deploy.h"

DEFINE_string(model_filename, "", "Path of det inference model");
DEFINE_string(params_filename, "", "Path of det inference params");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(model_type, "", "model type");
DEFINE_string(image_list, "", "Path of test image file");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
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
  engine_config.use_gpu = FLAGS_use_gpu;
  engine_config.gpu_id = FLAGS_gpu_id;
  engine_config.use_trt = FLAGS_use_trt;
  if (FLAGS_use_trt) {
    engine_config.precision = 0;
  }
  model->PaddleEngineInit(engine_config);

  // Mini-batch
  std::vector<std::string> image_paths;
  if (FLAGS_image_list != "") {
    std::ifstream inf(FLAGS_image_list);
    if (!inf) {
      std::cerr << "Fail to open file " << FLAGS_image_list << std::endl;
      return -1;
    }
    std::string image_path;
    while (getline(inf, image_path)) {
      image_paths.push_back(image_path);
    }
  }

  // infer
  std::vector<PaddleDeploy::Result> results;
  for (int i = 0; i < image_paths.size(); i += FLAGS_batch_size) {
    // Read image
    int im_vec_size =
        std::min(static_cast<int>(image_paths.size()), i + FLAGS_batch_size);
    std::vector<cv::Mat> im_vec(im_vec_size - i);
    #pragma omp parallel for num_threads(im_vec_size - i)
    for (int j = i; j < im_vec_size; ++j) {
      im_vec[j - i] = std::move(cv::imread(image_paths[j], 1));
    }

    model->Predict(im_vec, &results);

    std::cout << i / FLAGS_batch_size << " group -----" << std::endl;
    for (auto j = 0; j < results.size(); ++j) {
      std::cout << "Result for sample " << j << std::endl;
      std::cout << results[j] << std::endl;
    }
  }

  return 0;
}
