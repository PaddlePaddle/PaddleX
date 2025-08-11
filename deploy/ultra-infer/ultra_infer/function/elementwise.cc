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

#include "ultra_infer/function/elementwise.h"
#include "ultra_infer/function/eigen.h"
#include "ultra_infer/function/elementwise_base.h"
#include "ultra_infer/function/elementwise_functor.h"
#include "ultra_infer/utils/utils.h"
#include <algorithm>

namespace ultra_infer {
namespace function {

DEFINE_ELEMENTWISE_OP(Add);
DEFINE_ELEMENTWISE_OP(Multiply);
DEFINE_ELEMENTWISE_OP(Subtract);
DEFINE_ELEMENTWISE_OP(Divide);

void Add(const FDTensor &x, const FDTensor &y, FDTensor *out) {
  FD_VISIT_ALL_TYPES(x.dtype, "AddRawKernel",
                     ([&] { AddRawKernel<data_t>()(x, y, -1, out); }));
}

void Subtract(const FDTensor &x, const FDTensor &y, FDTensor *out) {
  FD_VISIT_ALL_TYPES(x.dtype, "SubtractRawKernel",
                     ([&] { SubtractRawKernel<data_t>()(x, y, -1, out); }));
}

void Multiply(const FDTensor &x, const FDTensor &y, FDTensor *out) {
  FD_VISIT_ALL_TYPES(x.dtype, "MultiplyRawKernel",
                     ([&] { MultiplyRawKernel<data_t>()(x, y, -1, out); }));
}

void Divide(const FDTensor &x, const FDTensor &y, FDTensor *out) {
  FD_VISIT_ALL_TYPES(x.dtype, "DivideRawKernel",
                     ([&] { DivideRawKernel<data_t>()(x, y, -1, out); }));
}

template <typename T> struct MaximumRawKernel {
  void operator()(const FDTensor &x, const FDTensor &y, int axis,
                  FDTensor *out) {
    ElementwiseCompute<MaximumFunctor<T>, T>(x, y, axis, MaximumFunctor<T>(),
                                             out);
  }
};

void Maximum(const FDTensor &x, const FDTensor &y, FDTensor *out) {
  FD_VISIT_ALL_TYPES(x.dtype, "MaximumRawKernel",
                     ([&] { MaximumRawKernel<data_t>()(x, y, -1, out); }));
}

} // namespace function

FDTensor operator+(const FDTensor &x, const FDTensor &y) {
  FDTensor out;
  function::Add(x, y, &out);
  return out;
}

FDTensor operator-(const FDTensor &x, const FDTensor &y) {
  FDTensor out;
  function::Subtract(x, y, &out);
  return out;
}

FDTensor operator*(const FDTensor &x, const FDTensor &y) {
  FDTensor out;
  function::Multiply(x, y, &out);
  return out;
}

FDTensor operator/(const FDTensor &x, const FDTensor &y) {
  FDTensor out;
  function::Divide(x, y, &out);
  return out;
}

#define INSTANTIATE_OPERATOR(operation_type)                                   \
  template FDTensor operator operation_type(const FDTensor &x, bool y);        \
  template FDTensor operator operation_type(const FDTensor &x, uint8_t y);     \
  template FDTensor operator operation_type(const FDTensor &x, int16_t y);     \
  template FDTensor operator operation_type(const FDTensor &x, int y);         \
  template FDTensor operator operation_type(const FDTensor &x, int64_t y);     \
  template FDTensor operator operation_type(const FDTensor &x, float y);       \
  template FDTensor operator operation_type(const FDTensor &x, double y);      \
  template FDTensor operator operation_type(bool x, const FDTensor &y);        \
  template FDTensor operator operation_type(uint8_t x, const FDTensor &y);     \
  template FDTensor operator operation_type(int16_t x, const FDTensor &y);     \
  template FDTensor operator operation_type(int x, const FDTensor &y);         \
  template FDTensor operator operation_type(int64_t x, const FDTensor &y);     \
  template FDTensor operator operation_type(float x, const FDTensor &y);       \
  template FDTensor operator operation_type(double x, const FDTensor &y)

INSTANTIATE_OPERATOR(+);
INSTANTIATE_OPERATOR(-);
INSTANTIATE_OPERATOR(*);
INSTANTIATE_OPERATOR(/);

} // namespace ultra_infer
