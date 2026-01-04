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

#if defined(WITH_GPU)

#include <cuda.h>
#include <cuda_runtime_api.h>

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
postprocess_gpu(const std::vector<paddle::Tensor> &hm,
                const std::vector<paddle::Tensor> &reg,
                const std::vector<paddle::Tensor> &height,
                const std::vector<paddle::Tensor> &dim,
                const std::vector<paddle::Tensor> &vel,
                const std::vector<paddle::Tensor> &rot,
                const std::vector<float> &voxel_size,
                const std::vector<float> &point_cloud_range,
                const std::vector<float> &post_center_range,
                const std::vector<int> &num_classes, const int down_ratio,
                const float score_threshold, const float nms_iou_threshold,
                const int nms_pre_max_size, const int nms_post_max_size,
                const bool with_velocity);

std::vector<paddle::Tensor>
centerpoint_postprocess(const std::vector<paddle::Tensor> &hm,
                        const std::vector<paddle::Tensor> &reg,
                        const std::vector<paddle::Tensor> &height,
                        const std::vector<paddle::Tensor> &dim,
                        const std::vector<paddle::Tensor> &vel,
                        const std::vector<paddle::Tensor> &rot,
                        const std::vector<float> &voxel_size,
                        const std::vector<float> &point_cloud_range,
                        const std::vector<float> &post_center_range,
                        const std::vector<int> &num_classes,
                        const int down_ratio, const float score_threshold,
                        const float nms_iou_threshold,
                        const int nms_pre_max_size, const int nms_post_max_size,
                        const bool with_velocity) {
  if (hm[0].is_gpu()) {
    return postprocess_gpu(hm, reg, height, dim, vel, rot, voxel_size,
                           point_cloud_range, post_center_range, num_classes,
                           down_ratio, score_threshold, nms_iou_threshold,
                           nms_pre_max_size, nms_post_max_size, with_velocity);
  } else {
    PD_THROW("Unsupported device type for centerpoint postprocess "
             "operator.");
  }
}

std::vector<std::vector<int64_t>>
PostProcessInferShape(const std::vector<std::vector<int64_t>> &hm_shape,
                      const std::vector<std::vector<int64_t>> &reg_shape,
                      const std::vector<std::vector<int64_t>> &height_shape,
                      const std::vector<std::vector<int64_t>> &dim_shape,
                      const std::vector<std::vector<int64_t>> &vel_shape,
                      const std::vector<std::vector<int64_t>> &rot_shape,
                      const std::vector<float> &voxel_size,
                      const std::vector<float> &point_cloud_range,
                      const std::vector<float> &post_center_range,
                      const std::vector<int> &num_classes, const int down_ratio,
                      const float score_threshold,
                      const float nms_iou_threshold, const int nms_pre_max_size,
                      const int nms_post_max_size, const bool with_velocity) {
  if (with_velocity) {
    return {{-1, 9}, {-1}, {-1}};
  } else {
    return {{-1, 7}, {-1}, {-1}};
  }
}

std::vector<paddle::DataType>
PostProcessInferDtype(const std::vector<paddle::DataType> &hm_dtype,
                      const std::vector<paddle::DataType> &reg_dtype,
                      const std::vector<paddle::DataType> &height_dtype,
                      const std::vector<paddle::DataType> &dim_dtype,
                      const std::vector<paddle::DataType> &vel_dtype,
                      const std::vector<paddle::DataType> &rot_dtype) {
  return {reg_dtype[0], hm_dtype[0], paddle::DataType::INT64};
}

} // namespace paddle_custom_ops
} // namespace ultra_infer

PD_BUILD_OP(centerpoint_postprocess)
    .Inputs({paddle::Vec("HM"), paddle::Vec("REG"), paddle::Vec("HEIGHT"),
             paddle::Vec("DIM"), paddle::Vec("VEL"), paddle::Vec("ROT")})
    .Outputs({"BBOXES", "SCORES", "LABELS"})
    .SetKernelFn(
        PD_KERNEL(ultra_infer::paddle_custom_ops::centerpoint_postprocess))
    .Attrs({"voxel_size: std::vector<float>",
            "point_cloud_range: std::vector<float>",
            "post_center_range: std::vector<float>",
            "num_classes: std::vector<int>", "down_ratio: int",
            "score_threshold: float", "nms_iou_threshold: float",
            "nms_pre_max_size: int", "nms_post_max_size: int",
            "with_velocity: bool"})
    .SetInferShapeFn(
        PD_INFER_SHAPE(ultra_infer::paddle_custom_ops::PostProcessInferShape))
    .SetInferDtypeFn(
        PD_INFER_DTYPE(ultra_infer::paddle_custom_ops::PostProcessInferDtype));

#endif // WITH_GPU
