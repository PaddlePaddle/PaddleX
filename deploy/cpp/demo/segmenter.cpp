//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <chrono>  // NOLINT
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include "include/paddlex/paddlex.h"
#include "include/paddlex/visualize.h"

using namespace std::chrono;  // NOLINT

DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_bool(use_trt, false, "Infering with TensorRT");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_string(key, "", "key of encryption");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image list file");
DEFINE_string(save_dir, "output", "Path to save visualized image");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_int32(thread_num,
             omp_get_num_procs(),
             "Number of preprocessing threads");

int main(int argc, char** argv) {
  // 解析命令行参数
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_dir == "") {
    std::cerr << "--model_dir need to be defined" << std::endl;
    return -1;
  }
  if (FLAGS_image == "" & FLAGS_image_list == "") {
    std::cerr << "--image or --image_list need to be defined" << std::endl;
    return -1;
  }

  // 加载模型
  PaddleX::Model model;
  model.Init(FLAGS_model_dir,
             FLAGS_use_gpu,
             FLAGS_use_trt,
             FLAGS_gpu_id,
             FLAGS_key);
  int imgs = 1;
  // 进行预测
  if (FLAGS_image_list != "") {
    std::ifstream inf(FLAGS_image_list);
    if (!inf) {
      std::cerr << "Fail to open file " << FLAGS_image_list << std::endl;
      return -1;
    }
    std::string image_path;
    std::vector<std::string> image_paths;
    while (getline(inf, image_path)) {
      image_paths.push_back(image_path);
    }
    imgs = image_paths.size();
    for (int i = 0; i < image_paths.size(); i += FLAGS_batch_size) {
      int im_vec_size =
          std::min(static_cast<int>(image_paths.size()), i + FLAGS_batch_size);
      std::vector<cv::Mat> im_vec(im_vec_size - i);
      std::vector<PaddleX::SegResult> results(im_vec_size - i,
                                              PaddleX::SegResult());
      int thread_num = std::min(FLAGS_thread_num, im_vec_size - i);
      #pragma omp parallel for num_threads(thread_num)
      for (int j = i; j < im_vec_size; ++j) {
        im_vec[j - i] = std::move(cv::imread(image_paths[j], 1));
      }
      model.predict(im_vec, &results, thread_num);
      // 可视化
      for (int j = 0; j < im_vec_size - i; ++j) {
        cv::Mat vis_img =
            PaddleX::Visualize(im_vec[j], results[j], model.labels);
        std::string save_path =
            PaddleX::generate_save_path(FLAGS_save_dir, image_paths[i + j]);
        cv::imwrite(save_path, vis_img);
        std::cout << "Visualized output saved as " << save_path << std::endl;
      }
    }
  } else {
    PaddleX::SegResult result;
    cv::Mat im = cv::imread(FLAGS_image, 1);
    model.predict(im, &result);
    // 可视化
    cv::Mat vis_img = PaddleX::Visualize(im, result, model.labels);
    std::string save_path =
        PaddleX::generate_save_path(FLAGS_save_dir, FLAGS_image);
    cv::imwrite(save_path, vis_img);
    result.clear();
    std::cout << "Visualized output saved as " << save_path << std::endl;
  }
  return 0;
}
