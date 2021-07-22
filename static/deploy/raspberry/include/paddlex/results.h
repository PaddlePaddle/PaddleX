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

template <class T>
struct Mask {
  std::vector<T> data;
  std::vector<int> shape;
  void clear() {
    data.clear();
    shape.clear();
  }
};

struct Box {
  int category_id;
  std::string category;
  float score;
  std::vector<float> coordinate;
  Mask<float> mask;
};

class BaseResult {
 public:
  std::string type = "base";
};

class ClsResult : public BaseResult {
 public:
  int category_id;
  std::string category;
  float score;
  std::string type = "cls";
};

class DetResult : public BaseResult {
 public:
  std::vector<Box> boxes;
  int mask_resolution;
  std::string type = "det";
  void clear() { boxes.clear(); }
};

class SegResult : public BaseResult {
 public:
  Mask<int64_t> label_map;
  Mask<float> score_map;
  void clear() {
    label_map.clear();
    score_map.clear();
  }
};
}  // namespace PaddleX
