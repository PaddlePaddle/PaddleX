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

#include "ultra_infer/function/linspace.h"

namespace ultra_infer {
namespace function {

template <typename T>
void LinspaceKernel(double start, double end, int num, FDTensor *out) {
  FDASSERT(
      num > 0,
      "The num of linspace op should be larger than 0, but received num is %d",
      num);
  out->Allocate({num}, TypeToDataType<T>::dtype);
  T *out_data = reinterpret_cast<T *>(out->Data());
  if (num > 1) {
    // step should be of double type for all types
    double step = (static_cast<double>(end - start)) / (num - 1);
    int half_num = num / 2;
    for (int i = 0; i < num; ++i) {
      if (i < half_num) {
        out_data[i] = static_cast<T>(start + step * i);
      } else {
        out_data[i] = static_cast<T>(end - step * (num - i - 1));
      }
    }
  } else {
    out_data[0] = static_cast<T>(start);
  }
}

void Linspace(double start, double end, int num, FDTensor *out,
              FDDataType dtype) {
  FD_VISIT_INT_FLOAT_TYPES(dtype, "LinspaceKernel", ([&] {
                             LinspaceKernel<data_t>(start, end, num, out);
                           }));
}

} // namespace function
} // namespace ultra_infer
