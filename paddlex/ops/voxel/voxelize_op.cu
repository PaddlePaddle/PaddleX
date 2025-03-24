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

#include "paddle/extension.h"

#define CHECK_INPUT_CUDA(x)                                                    \
  PD_CHECK(x.is_gpu() || x.is_gpu_pinned(), #x " must be a GPU Tensor.")

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                \
       i += blockDim.x * gridDim.x)

template <typename T, typename T_int>
__global__ void init_num_point_grid(
    const T *points, const float point_cloud_range_x_min,
    const float point_cloud_range_y_min, const float point_cloud_range_z_min,
    const float voxel_size_x, const float voxel_size_y,
    const float voxel_size_z, const int grid_size_x, const int grid_size_y,
    const int grid_size_z, const int64_t num_points, const int num_point_dim,
    T_int *num_points_in_grid, int *points_valid) {
  int64_t point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx > num_points || point_idx == num_points) {
    return;
  }
  int coord_x =
      floor((points[point_idx * num_point_dim + 0] - point_cloud_range_x_min) /
            voxel_size_x);
  int coord_y =
      floor((points[point_idx * num_point_dim + 1] - point_cloud_range_y_min) /
            voxel_size_y);
  int coord_z =
      floor((points[point_idx * num_point_dim + 2] - point_cloud_range_z_min) /
            voxel_size_z);

  if (coord_x < 0 || coord_x > grid_size_x || coord_x == grid_size_x) {
    return;
  }
  if (coord_y < 0 || coord_y > grid_size_y || coord_y == grid_size_y) {
    return;
  }
  if (coord_z < 0 || coord_z > grid_size_z || coord_z == grid_size_z) {
    return;
  }

  int grid_idx =
      coord_z * grid_size_y * grid_size_x + coord_y * grid_size_x + coord_x;
  num_points_in_grid[grid_idx] = 0;
  points_valid[grid_idx] = num_points;
}

template <typename T, typename T_int>
__global__ void map_point_to_grid_kernel(
    const T *points, const float point_cloud_range_x_min,
    const float point_cloud_range_y_min, const float point_cloud_range_z_min,
    const float voxel_size_x, const float voxel_size_y,
    const float voxel_size_z, const int grid_size_x, const int grid_size_y,
    const int grid_size_z, const int64_t num_points, const int num_point_dim,
    const int max_num_points_in_voxel, T_int *points_to_grid_idx,
    T_int *points_to_num_idx, T_int *num_points_in_grid, int *points_valid) {
  int64_t point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx > num_points || point_idx == num_points) {
    return;
  }
  int coord_x =
      floor((points[point_idx * num_point_dim + 0] - point_cloud_range_x_min) /
            voxel_size_x);
  int coord_y =
      floor((points[point_idx * num_point_dim + 1] - point_cloud_range_y_min) /
            voxel_size_y);
  int coord_z =
      floor((points[point_idx * num_point_dim + 2] - point_cloud_range_z_min) /
            voxel_size_z);

  if (coord_x < 0 || coord_x > grid_size_x || coord_x == grid_size_x) {
    return;
  }
  if (coord_y < 0 || coord_y > grid_size_y || coord_y == grid_size_y) {
    return;
  }
  if (coord_z < 0 || coord_z > grid_size_z || coord_z == grid_size_z) {
    return;
  }

  int grid_idx =
      coord_z * grid_size_y * grid_size_x + coord_y * grid_size_x + coord_x;
  T_int num = atomicAdd(num_points_in_grid + grid_idx, 1);
  if (num < max_num_points_in_voxel) {
    points_to_num_idx[point_idx] = num;
    points_to_grid_idx[point_idx] = grid_idx;
    atomicMin(points_valid + grid_idx, static_cast<int>(point_idx));
  }
}

template <typename T_int>
__global__ void update_points_flag(const int *points_valid,
                                   const T_int *points_to_grid_idx,
                                   const int num_points, int *points_flag) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < num_points; i += gridDim.x * blockDim.x) {
    T_int grid_idx = points_to_grid_idx[i];
    if (grid_idx >= 0) {
      int id = points_valid[grid_idx];
      if (id != num_points && id == i) {
        points_flag[i] = 1;
      }
    }
  }
}

template <typename T_int>
__global__ void
get_voxel_idx_kernel(const int *points_flag, const T_int *points_to_grid_idx,
                     const int *points_flag_prefix_sum, const int num_points,
                     const int max_voxels, T_int *num_voxels,
                     T_int *grid_idx_to_voxel_idx) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < num_points; i += gridDim.x * blockDim.x) {
    if (points_flag[i] == 1) {
      T_int grid_idx = points_to_grid_idx[i];
      int num = points_flag_prefix_sum[i];
      if (num < max_voxels) {
        grid_idx_to_voxel_idx[grid_idx] = num;
      }
    }
    if (i == num_points - 1) {
      int num = points_flag_prefix_sum[i] + points_flag[i];
      if (num < max_voxels) {
        num_voxels[0] = num;
      } else {
        num_voxels[0] = max_voxels;
      }
    }
  }
}

template <typename T>
__global__ void init_voxels_kernel(const int64_t num, T *voxels) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > num || idx == num) {
    return;
  }
  voxels[idx] = static_cast<T>(0);
}

template <typename T, typename T_int>
__global__ void
assign_voxels_kernel(const T *points, const T_int *points_to_grid_idx,
                     const T_int *points_to_num_idx,
                     const T_int *grid_idx_to_voxel_idx,
                     const int64_t num_points, const int num_point_dim,
                     const int max_num_points_in_voxel, T *voxels) {
  int64_t point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx > num_points || point_idx == num_points) {
    return;
  }
  T_int grid_idx = points_to_grid_idx[point_idx];
  T_int num_idx = points_to_num_idx[point_idx];
  if (grid_idx > -1 && num_idx > -1) {
    T_int voxel_idx = grid_idx_to_voxel_idx[grid_idx];
    if (voxel_idx > -1) {
      for (int64_t i = 0; i < num_point_dim; ++i) {
        voxels[voxel_idx * max_num_points_in_voxel * num_point_dim +
               num_idx * num_point_dim + i] =
            points[point_idx * num_point_dim + i];
      }
    }
  }
}

template <typename T, typename T_int>
__global__ void
assign_coords_kernel(const T_int *grid_idx_to_voxel_idx,
                     const T_int *num_points_in_grid, const int num_grids,
                     const int grid_size_x, const int grid_size_y,
                     const int grid_size_z, const int max_num_points_in_voxel,
                     T *coords, T *num_points_per_voxel) {
  int64_t grid_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (grid_idx > num_grids || grid_idx == num_grids) {
    return;
  }
  T_int voxel_idx = grid_idx_to_voxel_idx[grid_idx];
  if (voxel_idx > -1) {
    T_int coord_z = grid_idx / grid_size_x / grid_size_y;
    T_int coord_y =
        (grid_idx - coord_z * grid_size_x * grid_size_y) / grid_size_x;
    T_int coord_x =
        grid_idx - coord_z * grid_size_x * grid_size_y - coord_y * grid_size_x;
    coords[voxel_idx * 3 + 0] = coord_z;
    coords[voxel_idx * 3 + 1] = coord_y;
    coords[voxel_idx * 3 + 2] = coord_x;
    num_points_per_voxel[voxel_idx] =
        min(num_points_in_grid[grid_idx], max_num_points_in_voxel);
  }
}

std::vector<paddle::Tensor>
hard_voxelize_cuda(const paddle::Tensor &points,
                   const std::vector<float> &voxel_size,
                   const std::vector<float> &point_cloud_range,
                   int max_num_points_in_voxel, int max_voxels) {
  // check device
  CHECK_INPUT_CUDA(points);

  int64_t num_points = points.shape()[0];
  int64_t num_point_dim = points.shape()[1];

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
  int num_grids = grid_size_x * grid_size_y * grid_size_z;

  auto voxels =
      paddle::empty({max_voxels, max_num_points_in_voxel, num_point_dim},
                    paddle::DataType::FLOAT32, paddle::GPUPlace());

  auto coords = paddle::full({max_voxels, 3}, 0, paddle::DataType::INT32,
                             paddle::GPUPlace());
  auto *coords_data = coords.data<int>();

  auto num_points_per_voxel = paddle::full(
      {max_voxels}, 0, paddle::DataType::INT32, paddle::GPUPlace());
  auto *num_points_per_voxel_data = num_points_per_voxel.data<int>();

  auto points_to_grid_idx = paddle::full(
      {num_points}, -1, paddle::DataType::INT32, paddle::GPUPlace());
  auto *points_to_grid_idx_data = points_to_grid_idx.data<int>();

  auto points_to_num_idx = paddle::full(
      {num_points}, -1, paddle::DataType::INT32, paddle::GPUPlace());
  auto *points_to_num_idx_data = points_to_num_idx.data<int>();

  auto num_points_in_grid =
      paddle::empty({grid_size_z, grid_size_y, grid_size_x},
                    paddle::DataType::INT32, paddle::GPUPlace());
  auto *num_points_in_grid_data = num_points_in_grid.data<int>();

  auto grid_idx_to_voxel_idx =
      paddle::full({grid_size_z, grid_size_y, grid_size_x}, -1,
                   paddle::DataType::INT32, paddle::GPUPlace());
  auto *grid_idx_to_voxel_idx_data = grid_idx_to_voxel_idx.data<int>();

  auto num_voxels =
      paddle::full({1}, 0, paddle::DataType::INT32, paddle::GPUPlace());
  auto *num_voxels_data = num_voxels.data<int>();

  auto points_valid =
      paddle::empty({grid_size_z, grid_size_y, grid_size_x},
                    paddle::DataType::INT32, paddle::GPUPlace());
  int *points_valid_data = points_valid.data<int>();
  auto points_flag = paddle::full({num_points}, 0, paddle::DataType::INT32,
                                  paddle::GPUPlace());

  // 1. Find the grid index for each point, compute the
  // number of points in each grid
  int64_t threads = 512;
  int64_t blocks = (num_points + threads - 1) / threads;

  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "init_num_point_grid", ([&] {
        init_num_point_grid<data_t, int>
            <<<blocks, threads, 0, points.stream()>>>(
                points.data<data_t>(), point_cloud_range_x_min,
                point_cloud_range_y_min, point_cloud_range_z_min, voxel_size_x,
                voxel_size_y, voxel_size_z, grid_size_x, grid_size_y,
                grid_size_z, num_points, num_point_dim, num_points_in_grid_data,
                points_valid_data);
      }));

  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "map_point_to_grid_kernel", ([&] {
        map_point_to_grid_kernel<data_t, int>
            <<<blocks, threads, 0, points.stream()>>>(
                points.data<data_t>(), point_cloud_range_x_min,
                point_cloud_range_y_min, point_cloud_range_z_min, voxel_size_x,
                voxel_size_y, voxel_size_z, grid_size_x, grid_size_y,
                grid_size_z, num_points, num_point_dim, max_num_points_in_voxel,
                points_to_grid_idx_data, points_to_num_idx_data,
                num_points_in_grid_data, points_valid_data);
      }));

  // 2. Find the number of non-zero voxels
  int *points_flag_data = points_flag.data<int>();

  threads = 512;
  blocks = (num_points + threads - 1) / threads;
  update_points_flag<int><<<blocks, threads, 0, points.stream()>>>(
      points_valid_data, points_to_grid_idx_data, num_points, points_flag_data);

  auto points_flag_prefix_sum =
      paddle::experimental::cumsum(points_flag, 0, false, true, false);
  int *points_flag_prefix_sum_data = points_flag_prefix_sum.data<int>();

  get_voxel_idx_kernel<int><<<blocks, threads, 0, points.stream()>>>(
      points_flag_data, points_to_grid_idx_data, points_flag_prefix_sum_data,
      num_points, max_voxels, num_voxels_data, grid_idx_to_voxel_idx_data);

  // 3. Store points to voxels coords and num_points_per_voxel
  int64_t num = max_voxels * max_num_points_in_voxel * num_point_dim;
  threads = 512;
  blocks = (num + threads - 1) / threads;
  PD_DISPATCH_FLOATING_TYPES(points.type(), "init_voxels_kernel", ([&] {
                               init_voxels_kernel<data_t>
                                   <<<blocks, threads, 0, points.stream()>>>(
                                       num, voxels.data<data_t>());
                             }));

  threads = 512;
  blocks = (num_points + threads - 1) / threads;
  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "assign_voxels_kernel", ([&] {
        assign_voxels_kernel<data_t, int>
            <<<blocks, threads, 0, points.stream()>>>(
                points.data<data_t>(), points_to_grid_idx_data,
                points_to_num_idx_data, grid_idx_to_voxel_idx_data, num_points,
                num_point_dim, max_num_points_in_voxel, voxels.data<data_t>());
      }));

  // 4. Store coords, num_points_per_voxel
  blocks = (num_grids + threads - 1) / threads;
  assign_coords_kernel<int><<<blocks, threads, 0, points.stream()>>>(
      grid_idx_to_voxel_idx_data, num_points_in_grid_data, num_grids,
      grid_size_x, grid_size_y, grid_size_z, max_num_points_in_voxel,
      coords_data, num_points_per_voxel_data);

  return {voxels, coords, num_points_per_voxel, num_voxels};
}
