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

#include "ultra_infer/function/cast.h"
#include <algorithm>

namespace ultra_infer {
namespace function {

template <typename InT, typename OutT> struct CastOpTransformFunctor {
  OutT operator()(InT in) const { return static_cast<OutT>(in); }
};

template <typename InT>
void CastKernel(const FDTensor &x, FDTensor *out, FDDataType output_dtype) {

  FD_VISIT_ALL_TYPES(output_dtype, "CastOpTransformFunctor", ([&] {
                       auto *in_begin = reinterpret_cast<const InT *>(x.Data());
                       auto *in_end = in_begin + x.Numel();
                       FDTensor out_tmp;
                       out_tmp.Allocate(x.Shape(), output_dtype);
                       auto *out_begin =
                           reinterpret_cast<data_t *>(out_tmp.Data());
                       std::transform(in_begin, in_end, out_begin,
                                      CastOpTransformFunctor<InT, data_t>());
                       *out = std::move(out_tmp);
                     }));
}

void Cast(const FDTensor &x, FDTensor *out, FDDataType output_dtype) {
  FD_VISIT_ALL_TYPES(x.dtype, "CastKernel",
                     ([&] { CastKernel<data_t>(x, out, output_dtype); }));
}

} // namespace function
} // namespace ultra_infer
