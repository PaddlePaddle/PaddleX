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

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "include/paddlex/paddlex.h"

DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_string(cfg_file, "", "Path of PaddelX model yml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image list file");
DEFINE_int32(thread_num, 1, "num of thread to infer");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_dir == "") {
    std::cerr << "--model_dir need to be defined" << std::endl;
    return -1;
  }
  if (FLAGS_cfg_file == "") {
    std::cerr << "--cfg_flie need to be defined" << std::endl;
    return -1;
  }
  if (FLAGS_image == "" & FLAGS_image_list == "") {
    std::cerr << "--image or --image_list need to be defined" << std::endl;
    return -1;
  }

  // load model
  PaddleX::Model model;
  model.Init(FLAGS_model_dir, FLAGS_cfg_file, FLAGS_thread_num);
  std::cout << "init is done" << std::endl;
  // predict
  if (FLAGS_image_list != "") {
    std::ifstream inf(FLAGS_image_list);
    if (!inf) {
      std::cerr << "Fail to open file " << FLAGS_image_list << std::endl;
      return -1;
    }
    std::string image_path;

    while (getline(inf, image_path)) {
      PaddleX::ClsResult result;
      cv::Mat im = cv::imread(image_path, 1);
      model.predict(im, &result);
      std::cout << "Predict label: " << result.category
                << ", label_id:" << result.category_id
                << ", score: " << result.score << std::endl;
    }
  } else {
    PaddleX::ClsResult result;
    cv::Mat im = cv::imread(FLAGS_image, 1);
    model.predict(im, &result);
    std::cout << "Predict label: " << result.category
              << ", label_id:" << result.category_id
              << ", score: " << result.score << std::endl;
  }

  return 0;
}
