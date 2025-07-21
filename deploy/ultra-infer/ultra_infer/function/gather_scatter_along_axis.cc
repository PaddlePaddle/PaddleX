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

#include "ultra_infer/function/gather_scatter_along_axis.h"
#include "ultra_infer/function/tile.h"

namespace ultra_infer {
namespace function {

class TensorAssign {
public:
  template <typename tensor_t>
  void operator()(tensor_t *self_data, tensor_t *src_data) const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

template <typename T, typename index_t = int64_t, bool is_scatter_like = true>
struct GatherScatterFunctor {
  template <typename func_t>
  void operator()(const FDTensor &x, int axis, const FDTensor &index,
                  FDTensor *result, const func_t &reduce_op) {
    if (index.Numel() == 0) {
      return;
    }
    result->Allocate(index.Shape(), x.Dtype());
    const T *x_data = reinterpret_cast<const T *>(x.Data());
    const index_t *index_data = reinterpret_cast<const index_t *>(index.Data());
    T *result_data = reinterpret_cast<T *>(result->Data());

    int64_t x_size = x.Numel();
    int64_t index_size = index.Numel();
    int64_t result_size = result->Numel();
    auto x_dims = x.Shape();
    auto index_dims = index.Shape();
    auto result_dims = result->Shape();
    if (x_size == 0 || result_size == 0 || index_size == 0) {
      FDASSERT(false, "zero size input found, self_size, result_size, "
                      "index_size cannot be 0");
      return;
    }
    int select_dim_size = index_dims[axis];
    // index matrix has different shape with self matrix or src matrix.
    int replaced_select_dim_size =
        is_scatter_like ? result_dims[axis] : x_dims[axis];
    int64_t inner_dim_size = 1;
    int64_t outer_dim_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
      inner_dim_size *= index_dims[i];
    }

    for (int i = axis + 1; i < index_dims.size(); i++) {
      outer_dim_size *= index_dims[i];
    }
    int64_t index_idx = 0;
    int64_t self_idx, src_idx;
    // N layer loop squeezed into 3 layers loop
    for (int64_t i = 0; i < inner_dim_size; i++) {
      for (int64_t j = 0; j < select_dim_size; j++) {
        for (int64_t k = 0; k < outer_dim_size; k++) {
          int64_t index = index_data[index_idx];
          // This index might out of bound of index matrix's index, so here
          // multiply the replaced_select_dim_size.
          int64_t replace_index = k + index * outer_dim_size +
                                  i * outer_dim_size * replaced_select_dim_size;

          self_idx = is_scatter_like ? replace_index : index_idx;
          src_idx = is_scatter_like ? index_idx : replace_index;

          reduce_op((T *)(result_data + self_idx), // NOLINT
                    (T *)(x_data + src_idx));      // NOLINT

          index_idx++;
        }
      }
    }
  }
};

template <typename T> struct GatherFunctor {
  void operator()(const FDTensor &x, int axis, const FDTensor &index,
                  FDTensor *result) {
    FD_VISIT_INT_TYPES(index.Dtype(), "GatherFunctor", [&]() {
      auto x_shape = x.Shape();
      auto index_shape = index.Shape();
      std::vector<int64_t> repeat_times(x_shape.size(), 1);
      for (int i = 0; i < x_shape.size(); ++i) {
        repeat_times[i] = x_shape[i] / index_shape[i];
      }
      repeat_times[axis] = 1;
      FDTensor gs_index;
      Tile(index, repeat_times, &gs_index);
      GatherScatterFunctor<T, data_t, /*is_scatter_like=*/false>()(
          x, axis, gs_index, result, tensor_assign);
    });
  }
};

void GatherAlongAxis(const FDTensor &x, const FDTensor &index, FDTensor *result,
                     int axis) {
  int rank = x.Shape().size();
  FDASSERT(axis >= -rank && axis < rank,
           "axis should be in range [-%d, %d - 1].", rank, rank - 1);
  if (axis < 0) {
    axis += rank;
  }
  FD_VISIT_ALL_TYPES(x.Dtype(), "GatherAlongAxis", [&]() {
    GatherFunctor<data_t>()(x, axis, index, result);
  });
}

} // namespace function
} // namespace ultra_infer
