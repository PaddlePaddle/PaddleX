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

#pragma once

#include "ultra_infer/core/fd_scalar.h"
#include "ultra_infer/core/fd_tensor.h"

namespace ultra_infer {

namespace function {

/** Execute the add operation for input FDTensors. *out = x + y.
    @param x The input tensor.
    @param y The input tensor.
    @param out The output tensor which stores the result.
*/
ULTRAINFER_DECL void Add(const FDTensor &x, const FDTensor &y, FDTensor *out);

/** Execute the subtract operation for input FDTensors.  *out = x - y.
    @param x The input tensor.
    @param y The input tensor.
    @param out The output tensor which stores the result.
*/
ULTRAINFER_DECL void Subtract(const FDTensor &x, const FDTensor &y,
                              FDTensor *out);

/** Execute the multiply operation for input FDTensors.  *out = x * y.
    @param x The input tensor.
    @param y The input tensor.
    @param out The output tensor which stores the result.
*/
ULTRAINFER_DECL void Multiply(const FDTensor &x, const FDTensor &y,
                              FDTensor *out);

/** Execute the divide operation for input FDTensors.  *out = x / y.
    @param x The input tensor.
    @param y The input tensor.
    @param out The output tensor which stores the result.
*/
ULTRAINFER_DECL void Divide(const FDTensor &x, const FDTensor &y,
                            FDTensor *out);

/** Execute the maximum operation for input FDTensors.  *out = max(x, y).
    @param x The input tensor.
    @param y The input tensor.
    @param out The output tensor which stores the result.
*/
ULTRAINFER_DECL void Maximum(const FDTensor &x, const FDTensor &y,
                             FDTensor *out);

} // namespace function

ULTRAINFER_DECL FDTensor operator+(const FDTensor &x, const FDTensor &y);

template <typename T> FDTensor operator+(const FDTensor &x, T y) {
  return x + FDTensor(Scalar(y));
}

template <typename T> FDTensor operator+(T x, const FDTensor &y) {
  return FDTensor(Scalar(x)) + y;
}

ULTRAINFER_DECL FDTensor operator-(const FDTensor &x, const FDTensor &y);

template <typename T> FDTensor operator-(const FDTensor &x, T y) {
  return x - FDTensor(Scalar(y));
}

template <typename T> FDTensor operator-(T x, const FDTensor &y) {
  return FDTensor(Scalar(x)) - y;
}

ULTRAINFER_DECL FDTensor operator*(const FDTensor &x, const FDTensor &y);

template <typename T> FDTensor operator*(const FDTensor &x, T y) {
  return x * FDTensor(Scalar(y));
}

template <typename T> FDTensor operator*(T x, const FDTensor &y) {
  return FDTensor(Scalar(x)) * y;
}

ULTRAINFER_DECL FDTensor operator/(const FDTensor &x, const FDTensor &y);

template <typename T> FDTensor operator/(const FDTensor &x, T y) {
  return x / FDTensor(Scalar(y));
}

template <typename T> FDTensor operator/(T x, const FDTensor &y) {
  return FDTensor(Scalar(x)) / y;
}

} // namespace ultra_infer
