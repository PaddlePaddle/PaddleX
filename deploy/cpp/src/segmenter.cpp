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

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "include/paddlex/paddlex.h"
#include "include/paddlex/visualize.h"

DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_bool(use_trt, false, "Infering with TensorRT");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image list file");
DEFINE_string(save_dir, "output", "Path to save visualized image");

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
  model.Init(FLAGS_model_dir, FLAGS_use_gpu, FLAGS_use_trt, FLAGS_gpu_id);

  auto colormap = PaddleX::GenerateColorMap(model.labels.size());
  // 进行预测
  if (FLAGS_image_list != "") {
    std::ifstream inf(FLAGS_image_list);
    if (!inf) {
      std::cerr << "Fail to open file " << FLAGS_image_list << std::endl;
      return -1;
    }
    std::string image_path;
    while (getline(inf, image_path)) {
      PaddleX::SegResult result;
      cv::Mat im = cv::imread(image_path, 1);
      model.predict(im, &result);
      // 可视化
      cv::Mat vis_img =
          PaddleX::VisualizeSeg(im, result, model.labels, colormap);
      std::string save_path =
          PaddleX::generate_save_path(FLAGS_save_dir, image_path);
      cv::imwrite(save_path, vis_img);
      result.clear();
      std::cout << "Visualized output saved as " << save_path << std::endl;
    }
  } else {
    PaddleX::SegResult result;
    cv::Mat im = cv::imread(FLAGS_image, 1);
    model.predict(im, &result);
    // 可视化
    cv::Mat vis_img = PaddleX::VisualizeSeg(im, result, model.labels, colormap);
    std::string save_path =
        PaddleX::generate_save_path(FLAGS_save_dir, FLAGS_image);
    cv::imwrite(save_path, vis_img);
    result.clear();
    std::cout << "Visualized output saved as " << save_path << std::endl;
  }

  return 0;
}
