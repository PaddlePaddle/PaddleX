//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(WITH_GPU)

#include "grid_sample_3d.h"

#include <vector>

#if defined(PADDLEINFERENCE_API_COMPAT_2_4_x)
#include "paddle/include/experimental/ext_all.h"
#elif defined(PADDLEINFERENCE_API_COMPAT_2_5_x)
#include "paddle/include/paddle/extension.h"
#else
#include "paddle/extension.h"
#endif

namespace ultra_infer {
namespace paddle_custom_ops {

std::vector<paddle::Tensor>
GridSample3DCUDAForward(const paddle::Tensor &x, const paddle::Tensor &grid,
                        const std::string &mode,
                        const std::string &padding_mode, bool align_corners);

std::vector<paddle::Tensor> GridSample3DForward(const paddle::Tensor &x,
                                                const paddle::Tensor &grid,
                                                const std::string &mode,
                                                const std::string &padding_mode,
                                                bool align_corners) {
  return GridSample3DCUDAForward(x, grid, mode, padding_mode, align_corners);
}

std::vector<paddle::Tensor>
GridSample3DCUDABackward(const paddle::Tensor &x, const paddle::Tensor &grid,
                         const paddle::Tensor &grad_out,
                         const std::string &mode,
                         const std::string &padding_mode, bool align_corners);

std::vector<paddle::Tensor>
GridSample3DBackward(const paddle::Tensor &x, const paddle::Tensor &grid,
                     const paddle::Tensor &grad_out, const std::string &mode,
                     const std::string &padding_mode, bool align_corners) {
  return GridSample3DCUDABackward(x, grid, grad_out, mode, padding_mode,
                                  align_corners);
}

std::vector<std::vector<int64_t>>
GridSample3DInferShape(std::vector<int64_t> x_shape,
                       std::vector<int64_t> grid_shape) {
  return {
      {x_shape[0], x_shape[1], grid_shape[1], grid_shape[2], grid_shape[3]}};
}

std::vector<std::vector<int64_t>>
GridSample3DInferBackShape(std::vector<int64_t> x_shape,
                           std::vector<int64_t> grid_shape) {
  return {x_shape};
}

std::vector<paddle::DataType>
GridSample3DInferDtype(paddle::DataType x_dtype, paddle::DataType grid_dtype) {
  return {x_dtype};
}

} // namespace paddle_custom_ops
} // namespace ultra_infer

PD_BUILD_OP(grid_sample_3d)
    .Inputs({"x", "grid"})
    .Attrs({"mode: std::string", "padding_mode: std::string",
            "align_corners: bool"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(ultra_infer::paddle_custom_ops::GridSample3DForward))
    .SetInferShapeFn(
        PD_INFER_SHAPE(ultra_infer::paddle_custom_ops::GridSample3DInferShape))
    .SetInferDtypeFn(
        PD_INFER_DTYPE(ultra_infer::paddle_custom_ops::GridSample3DInferDtype));

PD_BUILD_GRAD_OP(grid_sample_3d)
    .Inputs({"x", "grid", paddle::Grad("out")})
    .Attrs({"mode: std::string", "padding_mode: std::string",
            "align_corners: bool"})
    .Outputs({paddle::Grad("x")})
    .SetKernelFn(
        PD_KERNEL(ultra_infer::paddle_custom_ops::GridSample3DBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(
        ultra_infer::paddle_custom_ops::GridSample3DInferBackShape));

#endif
