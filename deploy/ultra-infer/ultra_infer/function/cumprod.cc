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

#include "ultra_infer/function/cumprod.h"

namespace ultra_infer {
namespace function {

void GetCumprodDimInfo(const std::vector<int64_t> &dim, int cumprod_dim,
                       size_t *outer_dim, size_t *mid_dim, size_t *inner_dim) {
  int dim_size = dim.size();
  FDASSERT(cumprod_dim >= -dim_size,
           "The input dim of CumprodOp should be larger than the opposite "
           "rank of input x which is %d. But received dim = %d",
           -dim_size, cumprod_dim);
  FDASSERT(cumprod_dim < dim_size,
           "The input dim of CumprodOp should be smaller than the "
           "rank of input x which is %d. But received dim = %d",
           dim_size, cumprod_dim);
  if (cumprod_dim < 0)
    cumprod_dim += dim_size;

  *outer_dim = 1;
  for (int i = 0; i < cumprod_dim; ++i) {
    *outer_dim *= dim[i];
  }
  *mid_dim = dim[cumprod_dim];
  *inner_dim = 1;
  for (int i = cumprod_dim + 1; i < dim_size; ++i) {
    *inner_dim *= dim[i];
  }
}

template <typename T>
void CumprodKernel(const FDTensor &x, FDTensor *out, int axis) {
  auto *x_data = reinterpret_cast<const T *>(x.Data());
  auto shape = x.Shape();

  size_t outer_dim = 1;
  size_t mid_dim = 1;
  size_t inner_dim = 1;
  GetCumprodDimInfo(shape, axis, &outer_dim, &mid_dim, &inner_dim);

  out->Allocate(x.Shape(), x.Dtype());
  auto *out_data = reinterpret_cast<T *>(out->Data());

  for (size_t i = 0; i < outer_dim; i++) {
    for (size_t j = 0; j < mid_dim; j++) {
      for (size_t k = 0; k < inner_dim; k++) {
        size_t pos = i * mid_dim * inner_dim + j * inner_dim + k;
        if (j == 0) {
          out_data[pos] = x_data[pos];
        } else {
          out_data[pos] = out_data[pos - inner_dim] * x_data[pos];
        }
      }
    }
  }
}

void Cumprod(const FDTensor &x, FDTensor *out, int axis) {
  FD_VISIT_INT_FLOAT_TYPES(x.dtype, "CumprodKernel",
                           ([&] { CumprodKernel<data_t>(x, out, axis); }));
}

} // namespace function
} // namespace ultra_infer
