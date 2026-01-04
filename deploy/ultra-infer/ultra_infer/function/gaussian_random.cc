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

#include "ultra_infer/function/gaussian_random.h"
#include <memory>
#include <random>
#include <utility>

namespace ultra_infer {
namespace function {

template <typename T>
void GaussianRandomKernel(const std::vector<int64_t> &shape, float mean,
                          float std, int seed, FDTensor *out) {
  std::normal_distribution<T> dist(mean, std);

  out->Allocate(shape, TypeToDataType<T>::dtype);
  int64_t size = out->Numel();
  T *data = reinterpret_cast<T *>(out->Data());
  std::mt19937_64 engine;
  engine.seed(seed);
  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(engine);
  }
}

void GaussianRandom(const std::vector<int64_t> &shape, FDTensor *out,
                    FDDataType dtype, float mean, float std, int seed) {
  FD_VISIT_FLOAT_TYPES(dtype, "GaussianRandomKernel", [&]() {
    GaussianRandomKernel<data_t>(shape, mean, std, seed, out);
  });
}

} // namespace function
} // namespace ultra_infer
