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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include "include/paddlex/paddlex.h"
#include "include/paddlex/visualize.h"


DEFINE_string(model_dir, "", "Path of openvino model xml file");
DEFINE_string(cfg_file, "", "Path of PaddleX model yaml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image list file");
DEFINE_string(save_dir, "", "Path to save visualized image");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_int32(thread_num, 1, "num of thread to infer");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir == "") {
    std::cerr << "--model_dir need to be defined" << std::endl;
    return -1;
  }
  if (FLAGS_cfg_file == "") {
    std::cerr << "--cfg_file need to be defined" << std::endl;
    return -1;
  }
  if (FLAGS_image == "" & FLAGS_image_list == "") {
    std::cerr << "--image or --image_list need to be defined" << std::endl;
    return -1;
  }

  // load model
  std::cout << "init start" << std::endl;
  PaddleX::Model model;
  model.Init(FLAGS_model_dir, FLAGS_cfg_file, FLAGS_thread_num);
  std::cout << "init done" << std::endl;
  int imgs = 1;
  auto colormap = PaddleX::GenerateColorMap(model.labels.size());
  if (FLAGS_image_list != "") {
    std::ifstream inf(FLAGS_image_list);
    if (!inf) {
    std::cerr << "Fail to open file " << FLAGS_image_list <<std::endl;
    return -1;
    }
    std::string image_path;

    while (getline(inf, image_path)) {
      PaddleX::SegResult result;
      cv::Mat im = cv::imread(image_path, 1);
      model.predict(im, &result);
      if (FLAGS_save_dir != "") {
      cv::Mat vis_img = PaddleX::Visualize(im, result, model.labels, colormap);
        std::string save_path =
          PaddleX::generate_save_path(FLAGS_save_dir, image_path);
        cv::imwrite(save_path, vis_img);
        std::cout << "Visualized output saved as " << save_path << std::endl;
      }
    }
  } else {
    PaddleX::SegResult result;
    cv::Mat im = cv::imread(FLAGS_image, 1);
    model.predict(im, &result);
    if (FLAGS_save_dir != "") {
      cv::Mat vis_img = PaddleX::Visualize(im, result, model.labels, colormap);
      std::string save_path =
          PaddleX::generate_save_path(FLAGS_save_dir, FLAGS_image);
      cv::imwrite(save_path, vis_img);
      std::cout << "Visualized` output saved as " << save_path << std::endl;
    }
    result.clear();
  }
  return 0;
}
