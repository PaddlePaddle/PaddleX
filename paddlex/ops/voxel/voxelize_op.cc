// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "paddle/extension.h"

template <typename T, typename T_int>
bool hard_voxelize_cpu_kernel(
    const T *points, const float point_cloud_range_x_min,
    const float point_cloud_range_y_min, const float point_cloud_range_z_min,
    const float voxel_size_x, const float voxel_size_y,
    const float voxel_size_z, const int grid_size_x, const int grid_size_y,
    const int grid_size_z, const int64_t num_points, const int num_point_dim,
    const int max_num_points_in_voxel, const int max_voxels, T *voxels,
    T_int *coords, T_int *num_points_per_voxel, T_int *grid_idx_to_voxel_idx,
    T_int *num_voxels) {
  std::fill(voxels,
            voxels + max_voxels * max_num_points_in_voxel * num_point_dim,
            static_cast<T>(0));

  num_voxels[0] = 0;
  int voxel_idx, grid_idx, curr_num_point;
  int coord_x, coord_y, coord_z;
  for (int point_idx = 0; point_idx < num_points; ++point_idx) {
    coord_x = floor(
        (points[point_idx * num_point_dim + 0] - point_cloud_range_x_min) /
        voxel_size_x);
    coord_y = floor(
        (points[point_idx * num_point_dim + 1] - point_cloud_range_y_min) /
        voxel_size_y);
    coord_z = floor(
        (points[point_idx * num_point_dim + 2] - point_cloud_range_z_min) /
        voxel_size_z);

    if (coord_x < 0 || coord_x > grid_size_x || coord_x == grid_size_x) {
      continue;
    }
    if (coord_y < 0 || coord_y > grid_size_y || coord_y == grid_size_y) {
      continue;
    }
    if (coord_z < 0 || coord_z > grid_size_z || coord_z == grid_size_z) {
      continue;
    }

    grid_idx =
        coord_z * grid_size_y * grid_size_x + coord_y * grid_size_x + coord_x;
    voxel_idx = grid_idx_to_voxel_idx[grid_idx];
    if (voxel_idx == -1) {
      voxel_idx = num_voxels[0];
      if (num_voxels[0] == max_voxels || num_voxels[0] > max_voxels) {
        continue;
      }
      num_voxels[0]++;
      grid_idx_to_voxel_idx[grid_idx] = voxel_idx;
      coords[voxel_idx * 3 + 0] = coord_z;
      coords[voxel_idx * 3 + 1] = coord_y;
      coords[voxel_idx * 3 + 2] = coord_x;
    }
    curr_num_point = num_points_per_voxel[voxel_idx];
    if (curr_num_point < max_num_points_in_voxel) {
      for (int j = 0; j < num_point_dim; ++j) {
        voxels[voxel_idx * max_num_points_in_voxel * num_point_dim +
               curr_num_point * num_point_dim + j] =
            points[point_idx * num_point_dim + j];
      }
      num_points_per_voxel[voxel_idx] = curr_num_point + 1;
    }
  }
  return true;
}

std::vector<paddle::Tensor>
hard_voxelize_cpu(const paddle::Tensor &points,
                  const std::vector<float> &voxel_size,
                  const std::vector<float> &point_cloud_range,
                  const int max_num_points_in_voxel, const int max_voxels) {
  auto num_points = points.shape()[0];
  auto num_point_dim = points.shape()[1];

  const float voxel_size_x = voxel_size[0];
  const float voxel_size_y = voxel_size[1];
  const float voxel_size_z = voxel_size[2];
  const float point_cloud_range_x_min = point_cloud_range[0];
  const float point_cloud_range_y_min = point_cloud_range[1];
  const float point_cloud_range_z_min = point_cloud_range[2];
  int grid_size_x = static_cast<int>(
      round((point_cloud_range[3] - point_cloud_range[0]) / voxel_size_x));
  int grid_size_y = static_cast<int>(
      round((point_cloud_range[4] - point_cloud_range[1]) / voxel_size_y));
  int grid_size_z = static_cast<int>(
      round((point_cloud_range[5] - point_cloud_range[2]) / voxel_size_z));

  auto voxels =
      paddle::empty({max_voxels, max_num_points_in_voxel, num_point_dim},
                    paddle::DataType::FLOAT32, paddle::CPUPlace());

  auto coords = paddle::full({max_voxels, 3}, 0, paddle::DataType::INT32,
                             paddle::CPUPlace());
  auto *coords_data = coords.data<int>();

  auto num_points_per_voxel = paddle::full(
      {max_voxels}, 0, paddle::DataType::INT32, paddle::CPUPlace());
  auto *num_points_per_voxel_data = num_points_per_voxel.data<int>();
  std::fill(num_points_per_voxel_data,
            num_points_per_voxel_data + num_points_per_voxel.size(),
            static_cast<int>(0));

  auto num_voxels =
      paddle::full({1}, 0, paddle::DataType::INT32, paddle::CPUPlace());
  auto *num_voxels_data = num_voxels.data<int>();

  auto grid_idx_to_voxel_idx =
      paddle::full({grid_size_z, grid_size_y, grid_size_x}, -1,
                   paddle::DataType::INT32, paddle::CPUPlace());
  auto *grid_idx_to_voxel_idx_data = grid_idx_to_voxel_idx.data<int>();

  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "hard_voxelize_cpu_kernel", ([&] {
        hard_voxelize_cpu_kernel<data_t, int>(
            points.data<data_t>(), point_cloud_range_x_min,
            point_cloud_range_y_min, point_cloud_range_z_min, voxel_size_x,
            voxel_size_y, voxel_size_z, grid_size_x, grid_size_y, grid_size_z,
            num_points, num_point_dim, max_num_points_in_voxel, max_voxels,
            voxels.data<data_t>(), coords_data, num_points_per_voxel_data,
            grid_idx_to_voxel_idx_data, num_voxels_data);
      }));

  return {voxels, coords, num_points_per_voxel, num_voxels};
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor>
hard_voxelize_cuda(const paddle::Tensor &points,
                   const std::vector<float> &voxel_size,
                   const std::vector<float> &point_cloud_range,
                   int max_num_points_in_voxel, int max_voxels);
#endif

std::vector<paddle::Tensor>
hard_voxelize(const paddle::Tensor &points,
              const std::vector<float> &voxel_size,
              const std::vector<float> &point_cloud_range,
              const int max_num_points_in_voxel, const int max_voxels) {
  if (points.is_cpu()) {
    return hard_voxelize_cpu(points, voxel_size, point_cloud_range,
                             max_num_points_in_voxel, max_voxels);
#ifdef PADDLE_WITH_CUDA
  } else if (points.is_gpu() || points.is_gpu_pinned()) {
    return hard_voxelize_cuda(points, voxel_size, point_cloud_range,
                              max_num_points_in_voxel, max_voxels);
#endif
  } else {
    PD_THROW("Unsupported device type for hard_voxelize "
             "operator.");
  }
}

std::vector<std::vector<int64_t>>
HardInferShape(std::vector<int64_t> points_shape,
               const std::vector<float> &voxel_size,
               const std::vector<float> &point_cloud_range,
               const int &max_num_points_in_voxel, const int &max_voxels) {
  return {{max_voxels, max_num_points_in_voxel, points_shape[1]},
          {max_voxels, 3},
          {max_voxels},
          {1}};
}

std::vector<paddle::DataType> HardInferDtype(paddle::DataType points_dtype) {
  return {points_dtype, paddle::DataType::INT32, paddle::DataType::INT32,
          paddle::DataType::INT32};
}

PD_BUILD_OP(hard_voxelize)
    .Inputs({"POINTS"})
    .Outputs({"VOXELS", "COORS", "NUM_POINTS_PER_VOXEL", "num_voxels"})
    .SetKernelFn(PD_KERNEL(hard_voxelize))
    .Attrs({"voxel_size: std::vector<float>",
            "point_cloud_range: std::vector<float>",
            "max_num_points_in_voxel: int", "max_voxels: int"})
    .SetInferShapeFn(PD_INFER_SHAPE(HardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(HardInferDtype));
