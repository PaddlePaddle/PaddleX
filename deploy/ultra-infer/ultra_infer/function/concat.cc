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

#include "ultra_infer/function/concat.h"

#include "ultra_infer/utils/utils.h"
#include <cstring>
#include <limits>
#include <set>
#include <sstream>

namespace ultra_infer {
namespace function {

std::vector<int64_t>
ComputeAndCheckConcatOutputShape(const std::vector<FDTensor> &input, int axis) {
  const size_t n = input.size();
  auto out_dims = input[0].shape;
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; ++i) {
    FDASSERT(input[i].shape.size() == out_dims.size(),
             "The shape of input[0] and input[%d] is expected to be equal. But "
             "received input[0]'s shape = %s, input[%d]'s shape = %s.",
             i, Str(out_dims).c_str(), i, Str(input[i].shape).c_str());
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis) {
        out_dims[axis] += input[i].shape[axis];
      } else {
        FDASSERT(
            input[0].shape[j] == input[i].shape[j],
            "The %d-th dimension of input[0] and input[%d] is expected to be "
            "equal."
            "But received input[0]'s shape = %s, input[%d]'s shape = %s.",
            j, i, Str(input[0].shape).c_str(), i, Str(input[i].shape).c_str());
      }
    }
  }
  return out_dims;
}

template <typename T> struct ConcatFunctor {
  void operator()(const std::vector<FDTensor> &input, int axis,
                  FDTensor *output) {
    size_t num = input.size();

    int64_t rows = 1;
    auto dim_0 = input[0].shape;
    for (int i = 0; i < axis; ++i) {
      rows *= dim_0[i];
    }
    int64_t out_rows = rows, out_cols = 0;

    std::vector<int64_t> input_cols(num);
    for (size_t i = 0; i < num; ++i) {
      int64_t t_cols = input[i].Numel() / rows;
      out_cols += t_cols;
      input_cols[i] = t_cols;
    }

    // computation
    T *output_data = reinterpret_cast<T *>(output->Data());
    int64_t col_idx = 0;
    for (size_t j = 0; j < num; ++j) {
      int64_t col_len = input_cols[j];
      const T *input_data = reinterpret_cast<const T *>(input[j].Data());
      for (int64_t k = 0; k < out_rows; ++k) {
        FDTensor::CopyBuffer(output_data + k * out_cols + col_idx,
                             input_data + k * col_len, sizeof(T) * col_len,
                             input[j].device, input[j].is_pinned_memory);
      }
      col_idx += col_len;
    }
  }
};

template <typename T>
void ConcatKernel(const std::vector<FDTensor> &input, FDTensor *output,
                  int axis) {
  auto output_shape = ComputeAndCheckConcatOutputShape(input, axis);
  FDTensor output_tmp;
  output_tmp.Resize(output_shape, TypeToDataType<T>::dtype, output->name,
                    input[0].device);

  ConcatFunctor<T> functor;
  functor(input, axis, &output_tmp);
  *output = std::move(output_tmp);
}

void Concat(const std::vector<FDTensor> &x, FDTensor *out, int axis) {
  FDASSERT(x.size() > 0,
           "The number of FDTensor array should be larger than 0, but the size "
           "of input is %d",
           x.size());
  int64_t rank = x[0].shape.size();
  FDASSERT(axis >= -rank && axis < rank,
           "The axis is expected to be in range of [%d, %d), but got %d", -rank,
           rank, axis);
  if (axis < 0) {
    axis += rank;
  }

  FD_VISIT_ALL_TYPES(x[0].dtype, "Concat",
                     ([&] { ConcatKernel<data_t>(x, out, axis); }));
}

} // namespace function
} // namespace ultra_infer
