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

#include "ultra_infer/function/full.h"
#include "ultra_infer/function/eigen.h"
#include <algorithm>

namespace ultra_infer {
namespace function {

template <typename T> void FullValue(FDTensor *tensor, const Scalar &val) {
  auto t = EigenVector<T>::Flatten(*tensor);
  auto &place = *EigenDeviceWrapper::GetInstance()->GetDevice();
  t.device(place) = t.constant(val.to<T>());
}

void Full(const Scalar &value, const std::vector<int64_t> &shape, FDTensor *out,
          FDDataType dtype) {
  FD_VISIT_ALL_TYPES(dtype, "Full", ([&] {
                       out->Allocate(shape, dtype);
                       FullValue<data_t>(out, value);
                     }));
}

void FullLike(const FDTensor &x, const Scalar &value, FDTensor *out,
              FDDataType dtype) {
  Full(value, x.Shape(), out, dtype);
}

} // namespace function
} // namespace ultra_infer
