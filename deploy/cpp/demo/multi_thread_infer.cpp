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

#include "model_deploy/common/include/multi_thread_model.h"

DEFINE_string(model_filename, "", "Path of det inference model");
DEFINE_string(params_filename, "", "Path of det inference params");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(model_type, "", "model type");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(gpu_id, "0", "GPU card id, example: 0,2,3");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_int32(thread_num, 1, "thread num of preprocessing");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<int> gpu_ids;
  std::stringstream gpu_ids_str(FLAGS_gpu_id);
  std::string temp;
  while (getline(gpu_ids_str, temp, ',')) {
    gpu_ids.push_back(std::stoi(temp));
  }

  for (auto gpu_id : gpu_ids) {
    std::cout << "gpu_id:" << gpu_id << std::endl;
  }

  std::cout << "start create model" << std::endl;
  // create model
  PaddleDeploy::MultiThreadModel model;
  if (!model.Init(FLAGS_model_type, FLAGS_cfg_file, gpu_ids.size())) {
    return -1;
  }

  // engine init
  PaddleDeploy::PaddleEngineConfig engine_config;
  engine_config.model_filename = FLAGS_model_filename;
  engine_config.params_filename = FLAGS_params_filename;
  engine_config.use_gpu = FLAGS_use_gpu;
  engine_config.max_batch_size = FLAGS_batch_size;
  // 如果开启gpu，gpu_ids为gpu的序号(可重复)。 比如 0,0,1 表示0卡上创建两个实例， 1卡上创建1个实例
  // 如果使用cpu，gpu_ids可用任意int数字， 数字个数代表线程数量。比如 0,0,0 表示创建三个实例
  if (!model.PaddleEngineInit(engine_config, gpu_ids)) {
    return -1;
  }

  // prepare data
  std::vector<cv::Mat> imgs;
  imgs.push_back(std::move(cv::imread(FLAGS_image)));

  std::vector<std::vector<PaddleDeploy::Result>> results(5);
  std::vector<std::future<bool>> futures(5);
  for(;;) {
    for(int i = 0; i < 5; i++) {
      futures[i] = model.AddPredictTask(imgs, &results[i]);
    }
    for(int i = 0; i < 5; i++) {
      futures[i].get();
      std::cout << i << " result:" << results[i][0] << std::endl;
    }

    /*
    std::vector<PaddleDeploy::Result> batch_results;
    std::vector<cv::Mat> batch_imgs;
    for(int i = 0; i < 5; i++) {
      batch_imgs.push_back(std::move(cv::imread(FLAGS_image)));
    }
    // 如果输入是大batch, 可用此接口自动拆分输入，均匀分配到各线程上运算(注意：会同步等待所有结果完成)
    model.Predict(batch_imgs, &batch_results);
    for(int i = 0; i < 5; i++) {
       std::cout << i << " batch_result:" << batch_results[i] << std::endl;
    }
    */
  }

  return 0;
}
