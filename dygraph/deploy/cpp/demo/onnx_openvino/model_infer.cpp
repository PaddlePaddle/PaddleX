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
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "model_deploy/common/include/paddle_deploy.h"

DEFINE_string(model_filename, "", "Path of det inference model");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(model_type, "", "model type");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image file");
DEFINE_string(device, "CPU", "Infering with VPU or CPU");
DEFINE_int32(batch_size, 1, "Batch size of infering");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);

  // create model
  std::shared_ptr<PaddleDeploy::Model> model =
        PaddleDeploy::CreateModel(FLAGS_model_type);

  // model init
  model->Init(FLAGS_cfg_file);

  // engine init
  PaddleDeploy::OpenVinoEngineConfig engine_config;
  engine_config.model_file_ = FLAGS_model_filename;
  engine_config.batch_size_ = 1;
  engine_config.device_ = FLAGS_device;
  model->OpenVinoEngineInit(engine_config);

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
}
