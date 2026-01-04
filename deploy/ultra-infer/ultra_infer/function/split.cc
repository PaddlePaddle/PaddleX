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

#include "ultra_infer/function/split.h"
#include "ultra_infer/utils/utils.h"
#include <cstring>

namespace ultra_infer {
namespace function {

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T> struct SplitFunctor {
public:
  void operator()(const FDTensor &input,
                  const std::vector<const FDTensor *> &ref_inputs, int axis,
                  std::vector<FDTensor> *outputs) {
    if (input.Numel() == 0) {
      return;
    }

    size_t num = outputs->size();

    int input_rows = 1;
    auto dim_0 = ref_inputs[0]->Shape();
    for (int i = 0; i < axis; ++i) {
      input_rows *= dim_0[i];
    }

    int input_cols = 0;

    std::vector<int64_t> output_cols(outputs->size());
    for (size_t i = 0; i < num; ++i) {
      int t_cols = ref_inputs[i]->Numel() / input_rows;
      input_cols += t_cols;
      output_cols[i] = t_cols;
    }

    // computation
    for (int k = 0; k < input_rows; ++k) {
      const T *src_ptr =
          reinterpret_cast<const T *>(input.Data()) + k * input_cols;
      int col_idx = 0;
      for (size_t j = 0; j < num; ++j) {
        int col_len = output_cols[j];
        auto *out_tensor = &(outputs->at(j));
        if (out_tensor != nullptr) {
          T *dst_ptr = reinterpret_cast<T *>(out_tensor->Data()) + k * col_len;
          std::memcpy(dst_ptr, src_ptr + col_idx, sizeof(T) * col_len);
        }
        col_idx += col_len;
      }
    }
  }
};

inline int GetSplitAxisValue(const FDTensor &x, int axis) {
  int rank = x.Shape().size();
  FDASSERT(axis >= -rank && axis < rank,
           "The axis is expected to be in range of [%d, %d), but got %d", -rank,
           rank, axis);
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis;
}

void CreateSplitOutputs(const FDTensor &x,
                        const std::vector<int> &sections_data,
                        std::vector<FDTensor> *outs, int axis) {
  axis = GetSplitAxisValue(x, axis);
  auto input_axis_dim = x.Shape().at(axis);
  std::vector<int> sections_vec;
  const int unknow_dim_val = -1;
  int unknow_dim_idx = -1;
  int num_of_unknow = 0;
  int sum_of_section = 0;

  for (size_t i = 0; i < sections_data.size(); ++i) {
    sections_vec.push_back(sections_data[i]);
    if (sections_data[i] == unknow_dim_val) {
      num_of_unknow++;
      unknow_dim_idx = i;
    } else {
      sum_of_section += sections_data[i];
    }
  }

  FDASSERT(num_of_unknow <= 1,
           "Only one dimension value of Attr(num_or_sections) "
           "in SplitOp can be -1. "
           "But received Attr(num_or_sections) = [%s].",
           Str(sections_data).c_str());
  if (unknow_dim_idx != -1) {
    // for example, input shape = [4 ,5], axis = 1, sections = [2, 3, -1].
    // input_axis_dim = 5, sum_of_sections = 5.
    // the following check will fail.
    FDASSERT(sum_of_section < input_axis_dim,
             "Sum of Attr(num_or_sections) other than unknown section "
             "must be less than the input's "
             "size "
             "along the split dimension. But received Attr(num_or_sections) "
             "= [%s], input(X)'s shape = [%s], Attr(dim) = %d.",
             Str(sections_data).c_str(), Str(x.Shape()).c_str(), axis);
    sections_vec[unknow_dim_idx] = input_axis_dim - sum_of_section;
  } else {
    FDASSERT(sum_of_section == input_axis_dim,
             "Sum of Attr(num_or_sections) must be equal to the input's "
             "size "
             "along the split dimension. But received Attr(num_or_sections)"
             " = [%s], input(X)'s shape = [%s], Attr(dim) = %d.",
             Str(sections_data).c_str(), Str(x.Shape()).c_str(), axis);
  }
  // fill out dims
  std::vector<std::vector<int64_t>> out_dims(sections_vec.size(), x.Shape());
  for (size_t i = 0; i < sections_vec.size(); ++i) {
    out_dims[i][axis] = sections_vec[i];
  }
  for (size_t i = 0; i < sections_vec.size(); ++i) {
    (*outs)[i].Allocate(out_dims[i], x.Dtype());
  }
}

template <typename T>
void SplitKernel(const FDTensor &x, const std::vector<int> &section,
                 std::vector<FDTensor> *outs, int axis) {
  size_t out_number = section.size();
  outs->resize(out_number);
  CreateSplitOutputs(x, section, outs, axis);

  std::vector<const FDTensor *> shape_refer;
  for (size_t j = 0; j < outs->size(); ++j) {
    shape_refer.emplace_back(&((*outs)[j]));
  }
  SplitFunctor<T> functor;
  functor(x, shape_refer, axis, outs);
}

void Split(const FDTensor &x, const std::vector<int> &num_or_sections,
           std::vector<FDTensor> *out, int axis) {
  FD_VISIT_ALL_TYPES(x.Dtype(), "Split", ([&] {
                       SplitKernel<data_t>(x, num_or_sections, out, axis);
                     }));
}

} // namespace function
} // namespace ultra_infer
