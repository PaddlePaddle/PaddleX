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

#include "ultra_infer/function/math.h"
#include "ultra_infer/function/eigen.h"
#include "ultra_infer/function/math_functor.h"

namespace ultra_infer {
namespace function {

#define DEFINE_ACTIVATION_KERNEL(name, functor_class)                          \
  template <typename T> void name##Kernel(const FDTensor &x, FDTensor *out) {  \
    functor_class<T> functor;                                                  \
    ActivationImpl<T, functor_class<T>>(x, out, functor);                      \
  }

template <typename T, typename Functor>
void ActivationImpl(const FDTensor &X, FDTensor *Out, const Functor &functor) {
  FDASSERT(Out != nullptr, "Output Out should not be nullptr");
  FDTensor out_tmp;
  auto x = EigenVector<T>::Flatten(X);
  out_tmp.Allocate(X.Shape(), X.Dtype());
  auto out = EigenVector<T>::Flatten(out_tmp);
  const auto &dev = *EigenDeviceWrapper::GetInstance()->GetDevice();
  functor(dev, x, out);
  *Out = std::move(out_tmp);
}

DEFINE_ACTIVATION_KERNEL(Sqrt, SqrtFunctor)
DEFINE_ACTIVATION_KERNEL(Log, LogFunctor)
DEFINE_ACTIVATION_KERNEL(Round, RoundFunctor)
DEFINE_ACTIVATION_KERNEL(Exp, ExpFunctor)
DEFINE_ACTIVATION_KERNEL(Abs, AbsFunctor)
DEFINE_ACTIVATION_KERNEL(Ceil, CeilFunctor)
DEFINE_ACTIVATION_KERNEL(Floor, FloorFunctor)

void Sqrt(const FDTensor &x, FDTensor *out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "SqrtKernel",
                       ([&] { SqrtKernel<data_t>(x, out); }));
}

void Log(const FDTensor &x, FDTensor *out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "LogKernel",
                       ([&] { LogKernel<data_t>(x, out); }));
}

void Round(const FDTensor &x, FDTensor *out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "RoundKernel",
                       ([&] { RoundKernel<data_t>(x, out); }));
}

void Exp(const FDTensor &x, FDTensor *out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "ExpKernel",
                       ([&] { ExpKernel<data_t>(x, out); }));
}

void Abs(const FDTensor &x, FDTensor *out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "AbsKernel",
                       ([&] { AbsKernel<data_t>(x, out); }));
}

void Ceil(const FDTensor &x, FDTensor *out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "CeilKernel",
                       ([&] { CeilKernel<data_t>(x, out); }));
}

void Floor(const FDTensor &x, FDTensor *out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "FloorKernel",
                       ([&] { FloorKernel<data_t>(x, out); }));
}

} // namespace function
} // namespace ultra_infer
