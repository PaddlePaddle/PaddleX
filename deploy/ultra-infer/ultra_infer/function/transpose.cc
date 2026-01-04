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

#include "ultra_infer/function/transpose.h"
#include "ultra_infer/function/eigen.h"
#include "ultra_infer/utils/utils.h"

namespace ultra_infer {
namespace function {
template <typename T> struct TransposeNormalKernel {
  void operator()(const FDTensor &in, FDTensor *out,
                  const std::vector<int64_t> &axis) {
    const int rank = axis.size();
    auto in_stride = GetStride(in.shape);
    auto out_stride = GetStride(out->shape);
    const T *in_ptr = reinterpret_cast<const T *>(in.Data());
    T *out_ptr = reinterpret_cast<T *>(out->Data());

    auto transpose_helper = [&](int64_t beg, int64_t end) {
      for (int64_t out_idx = beg; out_idx < end; ++out_idx) {
        int64_t in_idx = 0;
        int64_t tmp_idx = out_idx;
        // calculate the input index
        for (int i = 0; i < rank; ++i) {
          const int64_t coordinate = tmp_idx / out_stride[i];
          tmp_idx -= coordinate * out_stride[i];
          in_idx += coordinate * in_stride[axis[i]];
        }
        out_ptr[out_idx] = in_ptr[in_idx];
      }
    };
    transpose_helper(0, out->Numel());
  }
};

template <typename T, int Rank> struct TransposeKernelImpl {
  void operator()(const FDTensor &in, FDTensor *out,
                  const std::vector<int64_t> &axis) {
    Eigen::array<int, Rank> permute;
    for (int i = 0; i < Rank; i++) {
      permute[i] = axis[i];
    }

    auto &place = *EigenDeviceWrapper::GetInstance()->GetDevice();
    auto eigen_in = EigenTensor<T, Rank>::From(in);
    auto eigen_out = EigenTensor<T, Rank>::From(*out);
    eigen_out.device(place) = eigen_in.shuffle(permute);
  }
};

template <typename T>
void TransposeKernel(const FDTensor &x, FDTensor *out,
                     const std::vector<int64_t> &axis) {
  int rank = axis.size();
  switch (rank) {
  case 1:
    TransposeKernelImpl<T, 1> trans1;
    trans1(x, out, axis);
    break;
  case 2:
    TransposeKernelImpl<T, 2> trans2;
    trans2(x, out, axis);
    break;
  case 3:
    TransposeKernelImpl<T, 3> trans3;
    trans3(x, out, axis);
    break;
  case 4:
    TransposeKernelImpl<T, 4> trans4;
    trans4(x, out, axis);
    break;
  default:
    // for rank >= 4 situation
    TransposeNormalKernel<T> trans_normal;
    trans_normal(x, out, axis);
  }
}

void Transpose(const FDTensor &x, FDTensor *out,
               const std::vector<int64_t> &dims) {
  size_t dims_size = dims.size();
  FDASSERT(dims_size == x.shape.size(),
           "The input tensor's dimension should be equal to the dims's size. "
           "Expect dims size is %lu, but receive %lu.",
           x.shape.size(), dims_size);
  std::vector<int> count(dims_size, 0);
  for (size_t i = 0; i < dims_size; i++) {
    FDASSERT(dims[i] >= 0,
             "The dims should be greater than or equal to 0, but receive %lld.",
             dims[i]);
    FDASSERT(dims[i] < static_cast<int>(dims_size) && ++count[dims[i]] == 1,
             "Each element of Attribute axis should be a unique value range "
             "from 0 to (dims - 1), where the dims is the axis's size, unique "
             "value means this axis value can appear only once. ");
  }
  std::vector<int64_t> out_dims(dims_size);
  for (size_t i = 0; i < dims_size; i++) {
    out_dims[i] = x.shape[dims[i]];
  }

  // Note(zhoushunjie): The FDTensor out may equal to FDTensor x, so firstly we
  // use out_temp to get the transposed result, then we move the out_temp to
  // out.
  FDTensor out_temp;
  out_temp.Allocate(out_dims, x.dtype);
  FD_VISIT_ALL_TYPES(x.dtype, "TransposeKernel",
                     ([&] { TransposeKernel<data_t>(x, &out_temp, dims); }));
  *out = std::move(out_temp);
}

} // namespace function
} // namespace ultra_infer
