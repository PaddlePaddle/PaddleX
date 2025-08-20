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

#include "ultra_infer/function/tile.h"
#include "ultra_infer/function/eigen.h"

namespace ultra_infer {
namespace function {

template <typename T, int Rank>
void TileFunctor(const FDTensor &x,
                 const std::vector<int64_t> &origin_repeat_times,
                 FDTensor *out) {
  auto x_shape = x.Shape();
  auto repeat_times = origin_repeat_times;
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    FDASSERT(repeat_times[i] > 0,
             "All elements of the input 'repeat_times' "
             "for tile op must be positive integers, but "
             "the value received is %d.",
             repeat_times[i]);
  }
  if (repeat_times.size() < x_shape.size()) {
    int diff = x_shape.size() - repeat_times.size();
    repeat_times.insert(repeat_times.begin(), diff, 1);
  } else {
    int diff = repeat_times.size() - x_shape.size();
    x_shape.insert(x_shape.begin(), diff, 1);
  }
  FDASSERT(repeat_times.size() == x_shape.size(),
           "The rank (%d) of the input 'x' and the rank (%d) of the input "
           "'repeat_times' for tile op must match after promotion.",
           x_shape.size(), repeat_times.size());

  if (Rank == 0) {
    // Deep copy
    *out = x;
    return;
  }

  FDTensor out_tmp;
  Eigen::DSizes<Eigen::DenseIndex, Rank> bcast_dims;
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    bcast_dims[i] = repeat_times[i];
  }

  std::vector<int64_t> out_shape(x_shape);
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    out_shape[i] *= repeat_times[i];
  }

  out_tmp.Allocate(out_shape, x.Dtype());
  auto eigen_x = EigenTensor<T, Rank>::From(x, x_shape);
  auto eigen_out = EigenTensor<T, Rank>::From(out_tmp, out_shape);

  const auto &dev = *EigenDeviceWrapper::GetInstance()->GetDevice();
  eigen_out.device(dev) = eigen_x.broadcast(bcast_dims);

  *out = std::move(out_tmp);
}

template <typename T>
void TileKernel(const FDTensor &x, const std::vector<int64_t> &repeat_times,
                FDTensor *out) {
  auto rank = x.Shape().size();
  auto repeat_times_size = repeat_times.size();
  rank = (std::max)(rank, repeat_times_size);
  switch (rank) {
  case 0:
    *out = x;
    break;
  case 1:
    TileFunctor<T, 1>(x, repeat_times, out);
    break;
  case 2:
    TileFunctor<T, 2>(x, repeat_times, out);
    break;
  case 3:
    TileFunctor<T, 3>(x, repeat_times, out);
    break;
  case 4:
    TileFunctor<T, 4>(x, repeat_times, out);
    break;
  case 5:
    TileFunctor<T, 5>(x, repeat_times, out);
    break;
  case 6:
    TileFunctor<T, 6>(x, repeat_times, out);
    break;
  }
}

void Tile(const FDTensor &x, const std::vector<int64_t> &repeat_times,
          FDTensor *out) {
  FD_VISIT_ALL_TYPES(x.dtype, "TileKernel",
                     ([&] { TileKernel<data_t>(x, repeat_times, out); }));
}

} // namespace function
} // namespace ultra_infer
