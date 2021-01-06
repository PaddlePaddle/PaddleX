// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

struct READ_RESULT {
  // the number of scales
  int scale_num;
  // the pointer location relative to the scales
  float scales;
  // the ratio between from the pointer to the starting scale and
  // distance from the ending scale to the starting scale
  float ratio;
};

void creat_line_image(const std::vector<int64_t> &seg_image,
                      std::vector<unsigned char> *output);

void convert_1D_data(const std::vector<unsigned char> &line_image,
                     std::vector<unsigned int> *scale_data,
                     std::vector<unsigned int> *pointer_data);

void scale_mean_filtration(const std::vector<unsigned int> &scale_data,
                           std::vector<unsigned int> *scale_mean_data);

void get_meter_reader(const std::vector<unsigned int> &scale,
                      const std::vector<unsigned int> &pointer,
                      READ_RESULT *result);

void read_process(const std::vector<std::vector<int64_t>> &seg_image,
                  std::vector<READ_RESULT> *read_results,
                  const int thread_num);
