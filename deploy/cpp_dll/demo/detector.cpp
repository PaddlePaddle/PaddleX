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

namespace PaddleX {
std::string image;
void PredictImage(std::string image, PaddleX::DetResult result);

void Loadmodel() {
  std::string model_dir = "E:\\0608\\inference_model";
  std::string key = "";
  std::string image_list = "";
  std::string save_dir = "output";
  int gpu_id = 0;
  bool use_trt = 0;
  bool use_gpu = 1;

  // Load model and create a object detector

  PaddleX::DetResult result;
  PredictImage(image, result);
}

void PredictImage(std::string image, PaddleX::DetResult result) {
  image = "E:\\0608\\pic\\test.jpg";
  cv::Mat im = cv::imread(image, 1);

  // PaddleX::DetResult* result;
  std::string model_dir = "E:\\0608\\inference_model";
  std::string key = "";
  std::string image_list = "";

  int gpu_id = 0;
  bool use_trt = 0;
  bool use_gpu = 1;
  PaddleX::Model model;
  model.Init(model_dir, use_gpu, use_trt, gpu_id, key);
  model.predict(im, &result);
  for (int i = 0; i < result.boxes.size(); ++i) {
    std::cout << ", predict label: " << result.boxes[i].category
              << ", label_id:" << result.boxes[i].category_id
              << ", score: " << result.boxes[i].score
              << ", box(xmin, ymin, w, h):(" << result.boxes[i].coordinate[0]
              << ", " << result.boxes[i].coordinate[1] << ", "
              << result.boxes[i].coordinate[2] << ", "
              << result.boxes[i].coordinate[3] << ")" << std::endl;
  }

  // 可视化
  auto colormap = PaddleX::GenerateColorMap(model.labels.size());
  cv::Mat vis_img = PaddleX::Visualize(im, result, model.labels, colormap, 0.5);
  std::string save_dir = "output";
  std::string save_path = PaddleX::generate_save_path(save_dir, image);

  cv::imwrite(save_path, vis_img);
  // result->clear();
  std::cout << "Visualized output saved as " << save_path << std::endl;
}
}  // namespace PaddleX