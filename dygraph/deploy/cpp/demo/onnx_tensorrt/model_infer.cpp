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

DEFINE_string(model_file, "", "Path of inference model");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(model_type, "", "model type");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image file");
DEFINE_string(trt_cache_file, "", "Cache path to store optimized trt file");
DEFINE_bool(save_engine, false, "Save Trt Engine");
DEFINE_int32(gpu_id, 0, "GPU card id");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << "ParseCommandLineFlags:FLAGS_model_type="
            << FLAGS_model_type << " model_file="
            << FLAGS_model_file << std::endl;

  // create model
  std::shared_ptr<PaddleDeploy::Model> model =
        PaddleDeploy::CreateModel(FLAGS_model_type);
  if (!model) {
    std::cout << "no model_type: " << FLAGS_model_type
              << "  model=" << model << std::endl;
    return 0;
  }
  std::cout << "start model init " << std::endl;

  // model init
  model->Init(FLAGS_cfg_file);
  std::cout << "start engine init " << std::endl;

  // inference engine init
  TensorRTEngineConfig engine_config;
  engine_config.model_file_ = FLAGS_model_file;
  engine_config.cfg_file_ = FLAGS_cfg_file;
  engine_config.gpu_id_ = FLAGS_gpu_id;
  engine_config.save_engine_ = FLAGS_save_engine;
  engine_config.trt_cache_file_ = FLAGS_trt_cache_file;
  model->TensorRTInit(engine_config);

  // prepare data
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
  } else if (FLAGS_image != "") {
    image_paths.push_back(FLAGS_image);
  } else {
    std::cerr << "image_list or image should be defined" << std::endl;
    return -1;
  }

  std::cout << "start model predict " << image_paths.size() << std::endl;
  // infer
  std::vector<PaddleDeploy::Result> results;
  std::vector<cv::Mat> imgs;
  cv::Mat img;
  for (auto i = 0; i < image_paths.size(); ++i) {
    img = cv::imread(image_paths[i]);
    if (img.empty()) {
      std::cerr << "Fail to read image: " << i << std::endl;
      return -1;
    }
    imgs.clear();
    imgs.push_back(std::move(img));

    model->Predict(imgs, &results);

    std::cout << "image: " << image_paths[i] << std::endl;
    for (auto j = 0; j < results.size(); ++j) {
      std::cout << "Result for sample " << j << std::endl;
      std::cout << results[j] << std::endl;
    }
  }
  return 0;
}
