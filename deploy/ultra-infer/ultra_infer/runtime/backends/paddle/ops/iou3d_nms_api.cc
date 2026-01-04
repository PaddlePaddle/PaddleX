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

#if defined(PADDLEINFERENCE_API_COMPAT_2_4_x)
#include "paddle/include/experimental/ext_all.h"
#elif defined(PADDLEINFERENCE_API_COMPAT_2_5_x)
#include "paddle/include/paddle/extension.h"
#else
#include "paddle/extension.h"
#endif

#include <vector>

#include "iou3d_cpu.h"
#include "iou3d_nms.h"

namespace ultra_infer {
namespace paddle_custom_ops {

std::vector<std::vector<int64_t>>
NMSInferShape(std::vector<int64_t> boxes_shape) {
  int64_t keep_num = 1;
  return {{boxes_shape[0]}, {keep_num}};
}

std::vector<paddle::DataType> NMSInferDtype(paddle::DataType boxes_dtype) {
  return {paddle::DataType::INT64, paddle::DataType::INT64};
}

} // namespace paddle_custom_ops
} // namespace ultra_infer

#if defined(WITH_GPU)

PD_BUILD_OP(nms_gpu)
    .Inputs({"boxes"})
    .Outputs({"keep", "num_to_keep"})
    .Attrs({"nms_overlap_thresh: float"})
    .SetKernelFn(PD_KERNEL(ultra_infer::paddle_custom_ops::nms_gpu))
    .SetInferDtypeFn(
        PD_INFER_DTYPE(ultra_infer::paddle_custom_ops::NMSInferDtype))
    .SetInferShapeFn(
        PD_INFER_SHAPE(ultra_infer::paddle_custom_ops::NMSInferShape));

#endif
