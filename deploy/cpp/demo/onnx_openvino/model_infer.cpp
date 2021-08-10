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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "model_deploy/common/include/paddle_deploy.h"

DEFINE_string(xml_file, "", "Path of model xml file");
DEFINE_string(bin_file, "", "Path of model bin file");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(model_type, "", "model type");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(device, "CPU", "Infering with VPU or CPU");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);

  // create model
  PaddleDeploy::Model* model = PaddleDeploy::CreateModel(FLAGS_model_type);


  // model init
  model->Init(FLAGS_cfg_file);

  // engine init
  PaddleDeploy::OpenVinoEngineConfig engine_config;
  engine_config.xml_file_ = FLAGS_xml_file;
  engine_config.bin_file_ = FLAGS_bin_file;
  engine_config.batch_size_ = 1;
  engine_config.device_ = FLAGS_device;
  model->OpenVinoEngineInit(engine_config);

  // prepare data
  std::vector<cv::Mat> imgs;
  imgs.push_back(std::move(cv::imread(FLAGS_image)));

  // predict
  std::vector<PaddleDeploy::Result> results;
  model->Predict(imgs, &results, 1);

  std::cout << results[0] << std::endl;
  delete model;
  return 0;
}
