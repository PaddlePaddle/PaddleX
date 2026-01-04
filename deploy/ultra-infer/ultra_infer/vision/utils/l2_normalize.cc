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

#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace utils {

std::vector<float> L2Normalize(const std::vector<float> &values) {
  size_t num_val = values.size();
  if (num_val == 0) {
    return {};
  }
  std::vector<float> norm;
  float l2_sum_val = 0.f;
  for (size_t i = 0; i < num_val; ++i) {
    l2_sum_val += (values[i] * values[i]);
  }
  float l2_sum_sqrt = std::sqrt(l2_sum_val);
  norm.resize(num_val);
  for (size_t i = 0; i < num_val; ++i) {
    norm[i] = values[i] / l2_sum_sqrt;
  }
  return norm;
}

} // namespace utils
} // namespace vision
} // namespace ultra_infer
