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

#include "model_deploy/common/include/multi_gpu_model.h"

DEFINE_string(model_filename, "", "Path of det inference model");
DEFINE_string(params_filename, "", "Path of det inference params");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(model_type, "", "model type");
DEFINE_string(image_list, "", "Path of test image file");
DEFINE_string(gpu_id, "0", "GPU card id, example: 0,2,3");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_int32(thread_num, 1, "thread num of preprocessing");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << "ParseCommandLineFlags:FLAGS_model_type="
            << FLAGS_model_type << " model_filename="
            << FLAGS_model_filename << std::endl;

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
  PaddleDeploy::MultiGPUModel model;
  if (!model.Init(FLAGS_model_type, FLAGS_cfg_file, gpu_ids.size())) {
    return -1;
  }

  if (!model.PaddleEngineInit(FLAGS_model_filename,
                              FLAGS_params_filename,
                              gpu_ids)) {
    return -1;
  }
  // Mini-batch
  if (FLAGS_image_list == "") {
    std::cerr << "image_list should be defined" << std::endl;
    return -1;
  }
  std::vector<std::string> image_paths;
  std::ifstream inf(FLAGS_image_list);
  if (!inf) {
    std::cerr << "Fail to open file " << FLAGS_image_list << std::endl;
    return -1;
  }
  std::string image_path;
  while (getline(inf, image_path)) {
    image_paths.push_back(image_path);
  }

  std::cout << "start model predict " << image_paths.size() << std::endl;
  // infer
  for (int i = 0; i < image_paths.size(); i += FLAGS_batch_size) {
    // Read image
    int im_vec_size =
        std::min(static_cast<int>(image_paths.size()), i + FLAGS_batch_size);
    std::vector<cv::Mat> im_vec(im_vec_size - i);
    #pragma omp parallel for num_threads(im_vec_size - i)
    for (int j = i; j < im_vec_size; ++j) {
      im_vec[j - i] = std::move(cv::imread(image_paths[j], 1));
    }
    model.Predict(im_vec, FLAGS_thread_num);
    std::cout << i / FLAGS_batch_size << " group" << std::endl;
    model.PrintResult();
  }
  return 0;
}
