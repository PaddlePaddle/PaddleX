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
#include <omp.h>
#include <memory>
#include <string>

#include <glog/logging.h>

#include "common/include/model_factory.h"

DEFINE_string(model_filename, "", "Path of det inference model");
DEFINE_string(params_filename, "", "Path of det inference params");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(model_type, "", "model type");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image file");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_bool(use_mkl, true, "Infering with mkl");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_int32(thread_num, 1, "thread num of infering");
DEFINE_string(toolkit, "det", "Type of PaddleToolKit");
DEFINE_bool(use_cpu_nms, false, "whether postprocess with NMS");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << "ParseCommandLineFlags:FLAGS_model_type=" << FLAGS_model_type
            << " model_filename=" << FLAGS_model_filename << std::endl;

  // create model
  std::shared_ptr<PaddleDeploy::Model> model =
      PaddleDeploy::ModelFactory::CreateObject(FLAGS_model_type);
  if (!model) {
    std::cout << "no model_type: " << FLAGS_model_type << "  model=" << model
              << std::endl;
    return 0;
  }
  std::cout << "start model init " << std::endl;

  // model init
  if (!model->Init(FLAGS_cfg_file, FLAGS_use_cpu_nms)) {
    std::cerr << "model Init error" << std::endl;
    return 0;
  }
  std::cout << "start engine init " << std::endl;

  // inference engine int
  if (!model->PaddleEngineInit(FLAGS_model_filename, FLAGS_params_filename,
                               FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_use_mkl)) {
    std::cerr << "Paddle Engine Init error" << std::endl;
    return 0;
  }

  // read image
  std::vector<cv::Mat> imgs;
  cv::Mat img;
  img = cv::imread(FLAGS_image, 1);
  imgs.push_back(std::move(img));

  std::cout << "start model predict " << std::endl;
  // infer
  if (!model->Predict(imgs)) {
    return 0;
  }
  // model->Predict(imgs, FLAGS_batch_size, FLAGS_thread_num);
  model->PrintResult();

  model->Predict(imgs, FLAGS_thread_num);
  model->PrintResult();
}
