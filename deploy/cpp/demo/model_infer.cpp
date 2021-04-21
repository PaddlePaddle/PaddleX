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

#include "common/include/multi_gpu_model.h"

DEFINE_string(model_filename, "", "Path of det inference model");
DEFINE_string(params_filename, "", "Path of det inference params");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(model_type, "", "model type");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image file");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_string(gpu_id, "0", "GPU card id, example: 0,2,3");
DEFINE_bool(use_mkl, true, "Infering with mkl");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_int32(thread_num, 1, "thread num of infering");

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
  bool init_result = model.Init(FLAGS_model_type, FLAGS_cfg_file,
                                FLAGS_model_filename, FLAGS_params_filename,
                                gpu_ids, FLAGS_use_gpu, FLAGS_use_mkl);

  if (!init_result) {
    std::cerr << "model init error " << std::endl;
    return -1;
  }

  // get image
  std::vector<cv::Mat> imgs;
  cv::Mat img;
  if (FLAGS_image_list != "") {
    std::ifstream inf(FLAGS_image_list);
    if (!inf) {
      std::cerr << "Fail to open file " << FLAGS_image_list << std::endl;
      return -1;
    }
    std::string path;
    while (getline(inf, path)) {
      img = cv::imread(path, 1);
      if (img.empty()) {
        std::cerr << "Fail image path:" << path << std::endl;
        return -1;
      }
      imgs.push_back(std::move(img));
    }
  } else if (FLAGS_image != "") {
    imgs.push_back(std::move(cv::imread(FLAGS_image, 1)));
  } else {
    std::cerr << "image_list or image should be defined" << std::endl;
    return -1;
  }

  std::cout << "start model predict " << imgs.size() << std::endl;
  //infer
  size_t batch_size = FLAGS_batch_size * gpu_ids.size();
  int img_start = 0;
  size_t imgs_size = imgs.size();
  while (img_start < imgs_size) {
    if (img_start + batch_size > imgs_size) {
      batch_size = imgs_size - img_start;
    }
    model.Predict(std::ref(imgs), FLAGS_thread_num,
                  img_start, img_start + batch_size);
    model.PrintResult();
    img_start += batch_size;
  }

  return 0;
}
