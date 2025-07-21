// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "ultra_infer/vision/visualize/visualize.h"

namespace ultra_infer {
namespace vision {

static std::vector<int> global_fd_vis_color_map = std::vector<int>();

std::vector<int> GenerateColorMap(int num_classes) {
  if (num_classes < 10) {
    num_classes = 10;
  }
  std::vector<int> color_map(num_classes * 3, 0);
  for (int i = 0; i < num_classes; ++i) {
    int j = 0;
    int lab = i;
    while (lab) {
      color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j));
      color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j));
      color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j));
      ++j;
      lab >>= 3;
    }
  }
  return color_map;
}

// This class will deprecated, please not use it
int Visualize::num_classes_ = 0;
std::vector<int> Visualize::color_map_ = std::vector<int>();

const std::vector<int> &Visualize::GetColorMap(int num_classes) {
  if (num_classes < num_classes_) {
    return color_map_;
  }
  num_classes_ = num_classes;
  std::vector<int>().swap(color_map_);
  color_map_.resize(3 * num_classes_, 0);
  for (int i = 0; i < num_classes_; ++i) {
    int j = 0;
    int lab = i;
    while (lab) {
      color_map_[i * 3] |= (((lab >> 0) & 1) << (7 - j));
      color_map_[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j));
      color_map_[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j));
      ++j;
      lab >>= 3;
    }
  }
  return color_map_;
}

} // namespace vision
} // namespace ultra_infer
