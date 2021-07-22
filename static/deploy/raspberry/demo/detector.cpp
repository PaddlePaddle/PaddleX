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

#include <gflags/gflags.h>

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

DEFINE_string(model_dir, "", "Path of openvino model xml file");
DEFINE_string(cfg_file, "", "Path of PaddleX model yaml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image list file");
DEFINE_int32(thread_num, 1, "num of thread to infer");
DEFINE_string(save_dir, "", "Path to save visualized image");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_double(threshold,
              0.5,
              "The minimum scores of target boxes which are shown");

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
  PaddleX::Model model;
  model.Init(FLAGS_model_dir, FLAGS_cfg_file, FLAGS_thread_num);

  int imgs = 1;
  auto colormap = PaddleX::GenerateColorMap(model.labels.size());
  // predict
  if (FLAGS_image_list != "") {
    std::ifstream inf(FLAGS_image_list);
    if (!inf) {
      std::cerr << "Fail to open file " << FLAGS_image_list << std::endl;
      return -1;
    }
    std::string image_path;

    while (getline(inf, image_path)) {
      PaddleX::DetResult result;
      cv::Mat im = cv::imread(image_path, 1);
      model.predict(im, &result);
      if (FLAGS_save_dir != "") {
        cv::Mat vis_img = PaddleX::Visualize(
          im, result, model.labels, colormap, FLAGS_threshold);
        std::string save_path =
          PaddleX::generate_save_path(FLAGS_save_dir, FLAGS_image);
        cv::imwrite(save_path, vis_img);
        std::cout << "Visualized output saved as " << save_path << std::endl;
      }
    }
  } else {
  PaddleX::DetResult result;
  cv::Mat im = cv::imread(FLAGS_image, 1);
  model.predict(im, &result);
  for (int i = 0; i < result.boxes.size(); ++i) {
      std::cout << "image file: " << FLAGS_image << std::endl;
      std::cout << ", predict label: " << result.boxes[i].category
                << ", label_id:" << result.boxes[i].category_id
                << ", score: " << result.boxes[i].score
                << ", box(xmin, ymin, w, h):(" << result.boxes[i].coordinate[0]
                << ", " << result.boxes[i].coordinate[1] << ", "
                << result.boxes[i].coordinate[2] << ", "
                << result.boxes[i].coordinate[3] << ")" << std::endl;
    }
    if (FLAGS_save_dir != "") {
    // visualize
      cv::Mat vis_img = PaddleX::Visualize(
        im, result, model.labels, colormap, FLAGS_threshold);
      std::string save_path =
          PaddleX::generate_save_path(FLAGS_save_dir, FLAGS_image);
      cv::imwrite(save_path, vis_img);
      result.clear();
      std::cout << "Visualized output saved as " << save_path << std::endl;
    }
  }
  return 0;
}
