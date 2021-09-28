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
DEFINE_string(image, "", "Path of test image file");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_int32(gpu_id, 0, "GPU card id");

void infer(PaddleDeploy::Model* model, const std::vector<cv::Mat>& imgs,
           std::vector<PaddleDeploy::Result>* results, int thread_num) {
  model->Predict(imgs, results, thread_num);
}

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);

  // create model
  PaddleDeploy::Model* model1 = PaddleDeploy::CreateModel(FLAGS_model_type);
  PaddleDeploy::Model* model2 = PaddleDeploy::CreateModel(FLAGS_model_type);
  PaddleDeploy::Model* model3 = PaddleDeploy::CreateModel(FLAGS_model_type);

  // model init
  model1->Init(FLAGS_cfg_file);
  model2->Init(FLAGS_cfg_file);
  model3->Init(FLAGS_cfg_file);

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
  model1->PaddleEngineInit(engine_config);
  model2->PaddleEngineInit(engine_config);
  model3->PaddleEngineInit(engine_config);

  // 多线程需要用线程池或其它方式复用线程， 否则频繁开关线程可能导致推理引擎内存问题
  ThreadPool pool(3);
  pool.init();

  // prepare data
  std::vector<cv::Mat> imgs;
  imgs.push_back(std::move(cv::imread(FLAGS_image)));

  // predict
  std::vector<PaddleDeploy::Result> results1;
  std::vector<PaddleDeploy::Result> results2;
  std::vector<PaddleDeploy::Result> results3;

  auto future1 = pool.submit(infer, model1, ref(imgs), &results1, 1);
  future1.get();

  auto future2 = pool.submit(infer, model2, ref(imgs), &results1, 1);
  future2.get();

  auto future3 = pool.submit(infer, model3, ref(imgs), &results1, 1);
  future3.get();

  // print result
  std::cout << "result1:" << results1[0] << std::endl;
  std::cout << "result2:" << results2[0] << std::endl;
  std::cout << "result3:" << results3[0] << std::endl;

  pool.shutdown();
  delete model1;
  delete model2;
  delete model3;
  return 0;
}
