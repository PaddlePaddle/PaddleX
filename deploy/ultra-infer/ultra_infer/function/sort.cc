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

#include "ultra_infer/function/sort.h"
#include "ultra_infer/function/eigen.h"
#include "ultra_infer/function/transpose.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ultra_infer {
namespace function {

template <typename T, typename Type>
static void FullSort(Type input_height, Type input_width, int input_dim,
                     const FDTensor *input, FDTensor *out, FDTensor *indices,
                     bool descending) {
  out->Allocate(input->Shape(), input->Dtype());
  indices->Allocate(input->Shape(), TypeToDataType<Type>::dtype);

  T *t_out = reinterpret_cast<T *>(out->Data());
  Type *t_indices = reinterpret_cast<Type *>(indices->Data());

  for (Type i = 0; i < input_height; ++i) {
    std::vector<std::pair<T, Type>> col_vec;
    col_vec.reserve(input_width);
    if (input_dim == 1) {
      auto e_input = EigenVector<T>::Flatten(*input);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.push_back(std::pair<T, Type>(e_input(j), j));
      }
    } else {
      auto e_input = EigenMatrix<T>::Reshape(*input, input_dim - 1);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.push_back(std::pair<T, Type>(e_input(i, j), j));
      }
    }
    std::sort(col_vec.begin(), col_vec.end(),
              [&](const std::pair<T, Type> &l, const std::pair<T, Type> &r) {
                if (descending)
                  return (std::isnan(static_cast<double>(l.first)) &&
                          !std::isnan(static_cast<double>(r.first))) ||
                         (l.first > r.first);
                else
                  return (!std::isnan(static_cast<double>(l.first)) &&
                          std::isnan(static_cast<double>(r.first))) ||
                         (l.first < r.first);
              });

    for (Type j = 0; j < input_width; ++j) {
      t_out[i * input_width + j] = col_vec[j].first;
      t_indices[i * input_width + j] = col_vec[j].second;
    }
  }
}

template <typename T>
void SortKernel(const FDTensor &x, FDTensor *out, FDTensor *indices,
                FDDataType indices_type, bool descending, int axis) {
  auto input_shape = x.Shape();
  int rank = input_shape.size();
  axis = (axis < 0) ? (rank + axis) : axis;
  // Do full sort
  if (axis == -1 || axis + 1 == rank) {
    int64_t numel = x.Numel();
    int64_t input_width = input_shape[axis];
    int64_t input_height = numel / input_width;
    FD_VISIT_INT_TYPES(indices_type, "FullSort", ([&] {
                         FullSort<T, data_t>(input_height, input_width, rank,
                                             &x, out, indices, descending);
                       }));
  } else {
    // If not full sort do transpose
    std::vector<int64_t> trans;
    for (int i = 0; i < axis; i++) {
      trans.push_back(i);
    }
    trans.push_back(rank - 1);
    for (int i = axis + 1; i < rank - 1; i++) {
      trans.push_back(i);
    }
    trans.push_back(axis);

    FDTensor trans_inp;
    Transpose(x, &trans_inp, trans);
    int64_t numel = x.Numel();
    int64_t input_width = input_shape[axis];
    int64_t input_height = numel / input_width;
    FD_VISIT_INT_TYPES(indices_type, "FullSort", ([&] {
                         FullSort<T, data_t>(input_height, input_width, rank,
                                             &trans_inp, out, indices,
                                             descending);
                       }));
    // transpose back
    Transpose(*out, out, trans);
    Transpose(*indices, indices, trans);
  }
}

void Sort(const FDTensor &x, FDTensor *out, FDTensor *indices, int axis,
          bool descending, FDDataType indices_type) {
  FD_VISIT_INT_FLOAT_TYPES(x.dtype, "SortKernel", ([&] {
                             SortKernel<data_t>(x, out, indices, indices_type,
                                                descending, axis);
                           }));
}

} // namespace function
} // namespace ultra_infer
