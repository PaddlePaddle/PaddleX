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

#include "ultra_infer/function/softmax.h"

#include <cstdlib>

#include "ultra_infer/function/eigen.h"
#include "ultra_infer/utils/axis_utils.h"
#include "ultra_infer/utils/utils.h"

namespace ultra_infer {
namespace function {
template <typename T> struct ValueClip {
  T operator()(const T &x) const {
    const T kThreshold = static_cast<T>(-64.);
    return x < kThreshold ? kThreshold : x;
  }
};

template <typename T> struct SoftmaxEigen {
  void operator()(const FDTensor &x, FDTensor *out, int axis_dim) {
    constexpr int kBatchDim = 0;
    constexpr int kClassDim = 1;
    constexpr int kAxisDim = 1;

    auto logits = EigenMatrix<T>::From(x);
    auto softmax = EigenMatrix<T>::From(*out);

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);
    const int num_remain = num_classes / axis_dim;
    Eigen::DSizes<int, 1> along_axis(kAxisDim);
    Eigen::DSizes<int, 2> batch_classes(batch_size, num_classes);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);
    Eigen::DSizes<int, 3> batch_one_remain(batch_size, 1, num_remain);
    Eigen::DSizes<int, 3> one_axis_one(1, axis_dim, 1);
    Eigen::DSizes<int, 2> one_axis(1, axis_dim);
    Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);

    const auto &dev = *EigenDeviceWrapper::GetInstance()->GetDevice();
    // For numerical stability, logits should be shifted by maximum number along
    // axis, calculate shifted_logits into softmax tensor for memory reuse.
    if (num_remain == 1) {
      // axis == -1, axis and class in same dimension, calculate along
      // class dimension directly for higher performance
      softmax.device(dev) = (logits - logits.maximum(along_axis)
                                          .eval()
                                          .reshape(batch_by_one)
                                          .broadcast(one_by_class))
                                .unaryExpr(ValueClip<T>());
    } else {
      // axis != -1, class dimension split into (axis, remain), max and sum
      // should be calculated along axis dimension
      softmax.device(dev) =
          (logits.reshape(batch_axis_remain) - logits.reshape(batch_axis_remain)
                                                   .maximum(along_axis)
                                                   .eval()
                                                   .reshape(batch_one_remain)
                                                   .broadcast(one_axis_one)
                                                   .reshape(batch_axis_remain))
              .reshape(batch_classes)
              .unaryExpr(ValueClip<T>());
    }
    softmax.device(dev) = softmax.exp();
    softmax.device(dev) = (softmax * softmax.reshape(batch_axis_remain)
                                         .sum(along_axis)
                                         .inverse()
                                         .eval()
                                         .broadcast(one_axis));
  }
};

template <typename T>
void SoftmaxFunctor(const FDTensor &x, FDTensor *out, int axis) {
  SoftmaxEigen<T>()(x, out, axis);
}

template <typename T>
void SoftmaxKernel(const FDTensor &x, FDTensor *out, int axis) {
  const int rank = x.shape.size();
  const int calc_axis = CanonicalAxis(axis, rank);
  int axis_dim = x.shape[calc_axis];
  out->Allocate(x.shape, x.dtype);
  if (out->Numel() == 0) {
    return;
  }
  const int n = SizeToAxis(calc_axis, x.shape);
  const int d = SizeFromAxis(calc_axis, x.shape);
  // Reshape to 2d tensor

  FDTensor x_2d, out_2d;
  x_2d.SetExternalData({n, d}, x.dtype, const_cast<void *>(x.Data()));
  out_2d.SetExternalData({n, d}, out->dtype, out->Data());

  SoftmaxFunctor<T>(x_2d, &out_2d, axis_dim);
}

void Softmax(const FDTensor &x, FDTensor *out, int axis) {
  FDASSERT(
      std::abs(axis) < x.shape.size(),
      "The absolute given axis should be smaller than the input's "
      "dimension. Expected absolute axis is smaller than %lu, but receive %d.",
      x.shape.size(), std::abs(axis));
  // Note(zhoushunjie): The FDTensor out may equal to FDTensor x, so firstly we
  // use out_temp to get the softmax result, then we move the out_temp to out.
  FDTensor out_tmp;
  FD_VISIT_FLOAT_TYPES(x.dtype, "SoftmaxKernel",
                       ([&] { SoftmaxKernel<data_t>(x, &out_tmp, axis); }));
  *out = std::move(out_tmp);
}
} // namespace function
} // namespace ultra_infer
