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

#include "ultra_infer/function/quantile.h"
#include "ultra_infer/core/fd_scalar.h"
#include "ultra_infer/function/cast.h"
#include "ultra_infer/function/concat.h"
#include "ultra_infer/function/elementwise.h"
#include "ultra_infer/function/gather_scatter_along_axis.h"
#include "ultra_infer/function/isfinite.h"
#include "ultra_infer/function/math.h"
#include "ultra_infer/function/reduce.h"
#include "ultra_infer/function/sort.h"
#include "ultra_infer/function/transpose.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ultra_infer {
namespace function {

template <typename T>
void QuantileKernel(const FDTensor &x, const std::vector<double> &q,
                    const std::vector<int> &axis, FDTensor *out) {
  FDASSERT(q.size() > 0, "q should not be empty.");
  FDASSERT(axis.size() > 0, "axis should not be empty.");
  std::vector<int64_t> axis_src;
  std::vector<int64_t> out_shape = x.Shape();
  int64_t rank = x.Shape().size();
  for (auto axis_single : axis) {
    FDASSERT(axis_single >= -rank && axis_single < rank,
             "The axis is expected to be in range of [%d, %d), but got %d",
             -rank, rank, axis_single);
    if (axis_single < 0) {
      axis_single += rank;
    }
    axis_src.push_back(axis_single);
    out_shape[axis_single] = 1;
  }
  std::vector<int64_t> axis_dst;
  for (int64_t i = 0; i < rank; ++i) {
    if (std::find(axis_src.begin(), axis_src.end(), i) == axis_src.end()) {
      axis_dst.push_back(i);
    }
  }
  axis_dst.insert(axis_dst.end(), axis_src.begin(), axis_src.end());
  FDTensor y;
  Transpose(x, &y, axis_dst);
  std::vector<int64_t> y_shape(rank - axis_src.size(), 0);
  y_shape.push_back(-1);
  y.Reshape({y_shape});

  int64_t target_axis = rank - 1;
  FDTensor mask, valid_counts, mask_any;
  IsNan(y, &mask);
  Any(mask, &mask_any, {target_axis}, true);
  bool *mask_data = reinterpret_cast<bool *>(mask.Data());
  std::transform(mask_data, mask_data + mask.Numel(), mask_data,
                 [](const bool &val) { return !val; });
  Cast(mask_any, &mask_any, FDDataType::FP64);
  Cast(mask, &mask, FDDataType::FP64);
  Sum(mask, &valid_counts, {target_axis}, true);

  FDTensor one_tensor(Scalar(static_cast<double>(1.0)));

  std::vector<FDTensor> indices;
  FDTensor last_index(Scalar(static_cast<double>(x.Shape()[target_axis])));
  for (auto q_num : q) {
    FDASSERT(q_num >= 0 && q_num <= 1, "q should be in range [0, 1]");
    FDTensor q_tensor(static_cast<double>(q_num));
    FDTensor index = q_tensor * (valid_counts - one_tensor);
    index = mask_any * last_index + (one_tensor - mask_any) * index;
    indices.push_back(index);
  }

  std::vector<FDTensor> outputs;
  FDTensor sorted_tensor, sorted_indices_tensor;
  Sort(y, &sorted_tensor, &sorted_indices_tensor, target_axis);
  Cast(sorted_tensor, &sorted_tensor, FDDataType::FP64);

  FDTensor indices_below, indices_upper;
  for (auto &&index : indices) {
    Floor(index, &indices_below);
    Ceil(index, &indices_upper);
    Cast(indices_below, &indices_below, FDDataType::INT32);
    Cast(indices_upper, &indices_upper, FDDataType::INT32);
    FDTensor tensor_below, tensor_upper;
    GatherAlongAxis(sorted_tensor, indices_below, &tensor_below, target_axis);
    GatherAlongAxis(sorted_tensor, indices_upper, &tensor_upper, target_axis);
    // Need to cast to FP64 to compute with index and tensor_upper
    Cast(indices_below, &indices_below, FDDataType::FP64);

    FDTensor weight = index - indices_below;
    FDTensor out = tensor_below + weight * (tensor_upper - tensor_below);
    out.Squeeze(target_axis);
    if (out.Dtype() != x.Dtype()) {
      Cast(out, &out, x.Dtype());
    }
    outputs.push_back(std::move(out));
  }
  if (outputs.size() > 1) {
    // Execute stack operation
    for (auto &output : outputs) {
      output.ExpandDim(0);
    }
    Concat(outputs, out, 0);
  } else {
    *out = std::move(outputs[0]);
  }
}

void Quantile(const FDTensor &x, const std::vector<double> &q,
              const std::vector<int> &axis, FDTensor *out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "QuantileKernel",
                       ([&] { QuantileKernel<data_t>(x, q, axis, out); }));
}

} // namespace function
} // namespace ultra_infer
