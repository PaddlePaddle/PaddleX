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

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace PaddleX {

/*
 * @brief
 * This class represents mask in instance segmentation tasks.
 * */
template <class T>
struct Mask {
  // raw data of mask
  std::vector<T> data;
  // the shape of mask
  std::vector<int> shape;
  void clear() {
    data.clear();
    shape.clear();
  }
};

/*
 * @brief
 * This class represents target box in detection or instance segmentation tasks.
 * */
struct Box {
  int category_id;
  // category label this box belongs to
  std::string category;
  // confidence score
  float score;
  std::vector<float> coordinate;
  Mask<int> mask;
};

/*
 * @brief
 * This class is prediction result based class.
 * */
class BaseResult {
 public:
  // model type
  std::string type = "base";
};

/*
 * @brief
 * This class represent classification result.
 * */
class ClsResult : public BaseResult {
 public:
  int category_id;
  std::string category;
  float score;
  std::string type = "cls";
};

/*
 * @brief
 * This class represent detection or instance segmentation result.
 * */
class DetResult : public BaseResult {
 public:
  // target boxes
  std::vector<Box> boxes;
  int mask_resolution;
  std::string type = "det";
  void clear() { boxes.clear(); }
};

/*
 * @brief
 * This class represent segmentation result.
 * */
class SegResult : public BaseResult {
 public:
  // represent label of each pixel on image matrix
  Mask<int64_t> label_map;
  // represent score of each pixel on image matrix
  Mask<float> score_map;
  std::string type = "seg";
  void clear() {
    label_map.clear();
    score_map.clear();
  }
};
}  // namespace PaddleX
