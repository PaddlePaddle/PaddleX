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

#include "ultra_infer/function/clip.h"
#include <algorithm>

namespace ultra_infer {
namespace function {

template <typename T> class ClipFunctor {
public:
  explicit ClipFunctor(const T min, const T max) : min_(min), max_(max) {}
  T operator()(const T x) const {
    return x < min_ ? min_ : x > max_ ? max_ : x;
  }

private:
  T min_;
  T max_;
};

template <typename T>
void ClipKernel(const FDTensor &x, double min, double max, FDTensor *out) {
  T max_ = static_cast<T>(max);
  T min_ = static_cast<T>(min);

  FDASSERT(min_ < max_,
           "max should be greater than or equal to min. But received min = %f, "
           "max = %f",
           static_cast<float>(min_), static_cast<float>(max_));
  FDTensor tmp;
  tmp.Allocate(x.Shape(), x.Dtype());
  const T *x_data = reinterpret_cast<const T *>(x.Data());

  int64_t numel = x.Numel();
  T *out_data = reinterpret_cast<T *>(tmp.Data());

  std::transform(x_data, x_data + numel, out_data, ClipFunctor<T>(min_, max_));
  *out = std::move(tmp);
}

void Clip(const FDTensor &x, double min, double max, FDTensor *out) {
  FD_VISIT_INT_FLOAT_TYPES(x.dtype, "ClipKernel",
                           ([&] { ClipKernel<data_t>(x, min, max, out); }));
}

} // namespace function
} // namespace ultra_infer
