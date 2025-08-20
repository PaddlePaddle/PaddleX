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

#include "ultra_infer/function/pad.h"

#include <cstdlib>

#include "ultra_infer/function/eigen.h"
#include "ultra_infer/utils/utils.h"

namespace ultra_infer {
namespace function {
template <typename T, int Rank> struct PadEigen {
  using Array = std::array<std::pair<int64_t, int64_t>, Rank>;
  using Array32Bit = std::array<std::pair<int, int>, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using InType32BitIndex =
      Eigen::TensorMap<Eigen::Tensor<const T, Rank, Eigen::RowMajor, int>,
                       Eigen::Aligned>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType32BitIndex =
      Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor, int>,
                       Eigen::Aligned>;

  static void Eval(const Eigen::DefaultDevice &dev, OutType out,
                   const InType &in, const Array &padding, const T value) {
    out.device(dev) = in.pad(padding, value);
  }

  static void Eval32(const Eigen::DefaultDevice &dev, OutType32BitIndex out,
                     const InType32BitIndex &in, const Array32Bit &padding,
                     const T value) {
    out.device(dev) = in.pad(padding, value);
  }
};

template <typename T, size_t D>
void PadFunction(const std::vector<int> &pads, const FDTensor &src, T pad_value,
                 FDTensor *out) {
  std::array<std::pair<int64_t, int64_t>, D> paddings;

  for (size_t i = 0; i < paddings.size(); ++i) {
    paddings[i].first = pads[i * 2];
    paddings[i].second = pads[i * 2 + 1];
  }

  auto src_tensor = EigenTensor<T, D>::From(src);
  auto out_tensor = EigenTensor<T, D>::From(*out);

  const auto &dev = *EigenDeviceWrapper::GetInstance()->GetDevice();
  PadEigen<T, D>::Eval(dev, out_tensor, src_tensor, paddings, pad_value);
}

template <typename T>
void PaddingFunctor(int rank, const std::vector<int> &pads, T pad_value,
                    const FDTensor &src, FDTensor *out) {
  switch (rank) {
  case 1:
    PadFunction<T, 1>(pads, src, pad_value, out);
    break;
  case 2:
    PadFunction<T, 2>(pads, src, pad_value, out);
    break;
  case 3:
    PadFunction<T, 3>(pads, src, pad_value, out);
    break;
  case 4:
    PadFunction<T, 4>(pads, src, pad_value, out);
    break;
  case 5:
    PadFunction<T, 5>(pads, src, pad_value, out);
    break;
  case 6:
    PadFunction<T, 6>(pads, src, pad_value, out);
    break;
  default:
    FDASSERT(
        false,
        "Pad only support tensors with no more than 6 dimensions currently.");
  }
}

template <typename T>
void PadKernel(const FDTensor &x, const std::vector<int> &paddings,
               const T &pad_value, FDTensor *out) {
  std::vector<int64_t> new_shape(x.shape.size());
  for (size_t i = 0; i < x.shape.size(); ++i) {
    new_shape[i] = x.shape[i] + paddings[2 * i] + paddings[2 * i + 1];
  }
  out->Allocate(new_shape, x.dtype);
  PaddingFunctor<T>(x.shape.size(), paddings, pad_value, x, out);
}

void Pad(const FDTensor &x, FDTensor *out, const std::vector<int> &pads,
         float value) {
  FDASSERT(pads.size() == x.shape.size() * 2,
           "Size of pads:%zu must be 2 times of rank:%zu.", pads.size(),
           x.shape.size());
  FDTensor out_tmp;
  FD_VISIT_ALL_TYPES(x.dtype, "PadKernel",
                     ([&] { PadKernel<data_t>(x, pads, value, &out_tmp); }));
  *out = std::move(out_tmp);
}

} // namespace function
} // namespace ultra_infer
