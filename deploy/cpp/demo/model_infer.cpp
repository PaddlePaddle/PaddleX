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
DEFINE_int32(thread_num, 1, "thread num of preprocessing");
DEFINE_int32(mkl_thread_num, 8, "thread num of mkldnn");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << "ParseCommandLineFlags:FLAGS_model_type="
            << FLAGS_model_type << " model_filename="
            << FLAGS_model_filename << std::endl;

  // create model
  std::shared_ptr<PaddleDeploy::Model> model =
        PaddleDeploy::ModelFactory::CreateObject(FLAGS_model_type);
  if (!model) {
    std::cout << "no model_type: " << FLAGS_model_type
              << "  model=" << model << std::endl;
    return 0;
  }
  std::cout << "start model init " << std::endl;

  // model init
  model->Init(FLAGS_cfg_file);
  std::cout << "start engine init " << std::endl;

  // inference engine in
  model->PaddleEngineInit(FLAGS_model_filename,
                          FLAGS_params_filename,
                          FLAGS_use_gpu,
                          FLAGS_gpu_id,
                          FLAGS_use_mkl,
                          FLAGS_mkl_thread_num);

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
  } else if (FLAGS_image != "") {
    image_paths.push_back(FLAGS_image);
  } else {
    std::cerr << "image_list or image should be defined" << std::endl;
    return -1;
  }

  std::cout << "start model predict " << image_paths.size() << std::endl;
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

    model->Predict(im_vec, &results, FLAGS_thread_num);
    std::cout << i / FLAGS_batch_size << " group" << std::endl;
    for (auto j = 0; j < results.size(); ++j) {
      std::cout << "Result for sample " << j << std::endl;
      std::cout << results[j] << std::endl;
    }
  }
  return 0;
}
