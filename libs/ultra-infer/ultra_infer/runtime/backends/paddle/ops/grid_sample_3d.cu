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

#include <cuda.h>

#include "grid_sample_3d.h"

#if defined(PADDLEINFERENCE_API_COMPAT_2_4_x)
#include "paddle/include/experimental/ext_all.h"
#elif defined(PADDLEINFERENCE_API_COMPAT_2_5_x)
#include "paddle/include/paddle/extension.h"
#else
#include "paddle/extension.h"
#endif

namespace ultra_infer {
namespace paddle_custom_ops {

#define CHECK_INPUT_GPU(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

static __forceinline__ __device__ bool
InBounds3D(int64_t d, int64_t h, int64_t w, int64_t D, int64_t H, int64_t W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                                \
  index_type _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;               \
  for (index_type i = _i_n_d_e_x; _i_n_d_e_x < (n);                            \
       _i_n_d_e_x += blockDim.x * gridDim.x, i = _i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)

template <typename T>
static __forceinline__ __device__ T Unnormalize(T coord, int size,
                                                bool align_corners) {
  if (align_corners) {
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    return ((coord + 1.f) * size - 1) / 2;
  }
}

template <typename T>
static __forceinline__ __device__ T ClipIndexes(T in, int max_value) {
  return min(static_cast<T>(max_value), max(in, static_cast<T>(0)));
}

template <typename T>
static __forceinline__ __device__ T ReflectIndexes(T in, int twice_low,
                                                   int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<T>(0);
  }
  T min = static_cast<T>(twice_low) / 2;
  T span = static_cast<T>(twice_high - twice_low) / 2;
  in = fabs(in - min);
  T extra = fmod(in, span);
  int flips = static_cast<int>(floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

template <typename T>
static __forceinline__ __device__ T ComputePositions(T coord, int size,
                                                     PaddingMode padding_mode,
                                                     bool align_corners) {
  coord = Unnormalize<T>(coord, size, align_corners);
  if (padding_mode == PaddingMode::border) {
    coord = ClipIndexes(coord, size - 1);
  } else if (padding_mode == PaddingMode::reflect) {
    if (align_corners) {
      coord = ReflectIndexes(coord, 0, 2 * (size - 1));
    } else {
      coord = ReflectIndexes(coord, -1, 2 * size - 1);
    }
    coord = ClipIndexes(coord, size - 1);
  }
  return coord;
}

template <typename T, typename index_t>
__global__ void
GridSample3DCudaKernel(const index_t nthreads, index_t out_c, index_t out_d,
                       index_t out_h, index_t out_w, index_t in_d, index_t in_h,
                       index_t in_w, const T *input, const T *grid, T *output,
                       const Mode interpolation_mode,
                       const PaddingMode padding_mode, bool align_corners) {
  // printf("size: %d, %d, %d, %d, %d, %d \n", out_c, out_d, out_w, out_h, in_d,
  // in_w);
  index_t inp_sW = 1;
  index_t inp_sH = in_w;
  index_t inp_sD = in_h * in_w;
  index_t inp_sC = in_d * inp_sD;
  index_t inp_sN = out_c * inp_sC;

  index_t grid_sCoor = 1;
  index_t grid_sW = 3;
  index_t grid_sH = out_w * grid_sW;
  index_t grid_sD = out_h * grid_sH;
  index_t grid_sN = out_d * grid_sD;

  index_t out_sW = 1;
  index_t out_sH = out_w;
  index_t out_sD = out_h * out_w;
  index_t out_sC = out_d * out_sD;
  index_t out_sN = out_c * out_sC;

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_w;
    const index_t h = (index / out_w) % out_h;
    const index_t d = (index / (out_h * out_w)) % out_d;
    const index_t n = index / (out_d * out_h * out_w);
    const index_t grid_offset =
        n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;
    // get the corresponding input x, y, z co-ordinates from grid
    T ix = grid[grid_offset];
    T iy = grid[grid_offset + grid_sCoor];
    T iz = grid[grid_offset + 2 * grid_sCoor];
    ix = ComputePositions(ix, in_w, padding_mode, align_corners);
    iy = ComputePositions(iy, in_h, padding_mode, align_corners);
    iz = ComputePositions(iz, in_d, padding_mode, align_corners);
    // printf("ix: %f, iy: %f, iz: %f \n", ix, iy, iz);
    if (interpolation_mode == Mode::bilinear) {
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      index_t ix_tnw = static_cast<index_t>(std::floor(ix));
      index_t iy_tnw = static_cast<index_t>(std::floor(iy));
      index_t iz_tnw = static_cast<index_t>(std::floor(iz));

      index_t ix_tne = ix_tnw + 1;
      index_t iy_tne = iy_tnw;
      index_t iz_tne = iz_tnw;

      index_t ix_tsw = ix_tnw;
      index_t iy_tsw = iy_tnw + 1;
      index_t iz_tsw = iz_tnw;

      index_t ix_tse = ix_tnw + 1;
      index_t iy_tse = iy_tnw + 1;
      index_t iz_tse = iz_tnw;

      index_t ix_bnw = ix_tnw;
      index_t iy_bnw = iy_tnw;
      index_t iz_bnw = iz_tnw + 1;

      index_t ix_bne = ix_tnw + 1;
      index_t iy_bne = iy_tnw;
      index_t iz_bne = iz_tnw + 1;

      index_t ix_bsw = ix_tnw;
      index_t iy_bsw = iy_tnw + 1;
      index_t iz_bsw = iz_tnw + 1;

      index_t ix_bse = ix_tnw + 1;
      index_t iy_bse = iy_tnw + 1;
      index_t iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      T tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
      T tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
      T tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
      T tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
      T bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
      T bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
      T bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
      T bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCDHW =
          output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (index_t c = 0; c < out_c;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        *out_ptr_NCDHW = static_cast<T>(0);
        if (InBounds3D(iz_tnw, iy_tnw, ix_tnw, in_d, in_h, in_w)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] *
              tnw;
        }
        if (InBounds3D(iz_tne, iy_tne, ix_tne, in_d, in_h, in_w)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] *
              tne;
        }
        if (InBounds3D(iz_tsw, iy_tsw, ix_tsw, in_d, in_h, in_w)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] *
              tsw;
        }
        if (InBounds3D(iz_tse, iy_tse, ix_tse, in_d, in_h, in_w)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] *
              tse;
        }
        if (InBounds3D(iz_bnw, iy_bnw, ix_bnw, in_d, in_h, in_w)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] *
              bnw;
        }
        if (InBounds3D(iz_bne, iy_bne, ix_bne, in_d, in_h, in_w)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] *
              bne;
        }
        if (InBounds3D(iz_bsw, iy_bsw, ix_bsw, in_d, in_h, in_w)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] *
              bsw;
        }
        if (InBounds3D(iz_bse, iy_bse, ix_bse, in_d, in_h, in_w)) {
          *out_ptr_NCDHW +=
              inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] *
              bse;
        }
      }
    } else if (interpolation_mode == Mode::nearest) {
      index_t ix_nearest = static_cast<index_t>(std::round(ix));
      index_t iy_nearest = static_cast<index_t>(std::round(iy));
      index_t iz_nearest = static_cast<index_t>(std::round(iz));

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCDHW =
          output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (index_t c = 0; c < out_c;
           ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        if (InBounds3D(iz_nearest, iy_nearest, ix_nearest, in_d, in_h, in_w)) {
          *out_ptr_NCDHW =
              inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH +
                         ix_nearest * inp_sW];
        } else {
          *out_ptr_NCDHW = static_cast<T>(0);
        }
      }
    }
  }
}

std::vector<paddle::Tensor>
GridSample3DCUDAForward(const paddle::Tensor &x, const paddle::Tensor &grid,
                        const std::string &mode,
                        const std::string &padding_mode, bool align_corners) {
  CHECK_INPUT_GPU(x);
  CHECK_INPUT_GPU(grid);
  PaddingMode enum_padding_mode;
  Mode enum_mode;
  if (padding_mode == "border") {
    enum_padding_mode = PaddingMode::border;
  } else if (padding_mode == "reflection") {
    enum_padding_mode = PaddingMode::reflect;
  } else {
    enum_padding_mode = PaddingMode::zeros;
  }

  if (mode == "nearest") {
    enum_mode = Mode::nearest;
  } else {
    enum_mode = Mode::bilinear;
  }
  const int n = grid.shape()[0];
  const int out_d = grid.shape()[1];
  const int out_h = grid.shape()[2];
  const int out_w = grid.shape()[3];
  const int c = x.shape()[1];
  const int in_d = x.shape()[2];
  const int in_h = x.shape()[3];
  const int in_w = x.shape()[4];

  auto output = paddle::full({n, c, out_d, out_h, out_w}, 0,
                             paddle::DataType::FLOAT32, paddle::GPUPlace());
  const int count = static_cast<int>(n * out_d * out_h * out_w);

  int max_threads_per_block = 512;
  int block_num = (count - 1) / max_threads_per_block + 1;
  // printf("size: %d, %d, %d, %d, %d, %d \n", n, c, out_d, out_h, count,
  // block_num);
  GridSample3DCudaKernel<float, int>
      <<<block_num, max_threads_per_block, 0, x.stream()>>>(
          count, c, out_d, out_h, out_w, in_d, in_h, in_w, x.data<float>(),
          grid.data<float>(), output.data<float>(), enum_mode,
          enum_padding_mode, align_corners);

  cudaError_t error_check;
  error_check = cudaGetLastError();
  if (error_check != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(error_check));
  }
  // printf("size: %d, %d, %d, %d, %d, %d \n", n, c, out_d, out_h, count,
  // block_num);
  return {output};
}

template <typename T>
static __forceinline__ __device__ T UnnormalizeWithMask(T coord, int size,
                                                        bool align_corners,
                                                        T *grad_in) {
  if (align_corners) {
    *grad_in = static_cast<T>(size - 1) / 2;
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    *grad_in = static_cast<T>(size) / 2;
    return ((coord + 1.f) * size - 1) / 2;
  }
}

template <typename T>
static __forceinline__ __device__ T ClipIndexesWithMask(T in, int clip_limit,
                                                        T *grad_in) {
  if (in <= static_cast<T>(0)) {
    *grad_in = static_cast<T>(0);
    return static_cast<T>(0);
  } else {
    T max = static_cast<T>(clip_limit - 1);
    if (in >= max) {
      *grad_in = static_cast<T>(0);
      return max;
    } else {
      *grad_in = static_cast<T>(1);
      return in;
    }
  }
}

template <typename T>
static __forceinline__ __device__ T ReflectIndexesWithMask(T in, int twice_low,
                                                           int twice_high,
                                                           T *grad_in) {
  if (twice_low == twice_high) {
    *grad_in = static_cast<T>(0);
    return static_cast<T>(0);
  }
  int grad_in_mult_;
  T min = static_cast<T>(twice_low) / 2;
  T span = static_cast<T>(twice_high - twice_low) / 2;
  in = in - min;
  if (in < static_cast<T>(0)) {
    grad_in_mult_ = -1;
    in = -in;
  } else {
    grad_in_mult_ = 1;
  }
  T extra = fmod(in, span);
  int flips = static_cast<int>(floor(in / span));
  if (flips % 2 == 0) {
    *grad_in = static_cast<T>(grad_in_mult_);
    return extra + min;
  } else {
    *grad_in = static_cast<T>(-grad_in_mult_);
    return span - extra + min;
  }
}

template <typename T>
static __forceinline__ __device__ T
ComputePositionsWithMask(T coord, int size, PaddingMode padding_mode,
                         bool align_corners, T *grad_in) {
  T grad_clip, grad_refl;
  coord = UnnormalizeWithMask<T>(coord, size, align_corners, grad_in);
  if (padding_mode == PaddingMode::border) {
    coord = ClipIndexesWithMask(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_clip;
  } else if (padding_mode == PaddingMode::reflect) {
    if (align_corners) {
      coord = ReflectIndexesWithMask(coord, 0, 2 * (size - 1), &grad_refl);
    } else {
      coord = ReflectIndexesWithMask(coord, -1, 2 * size - 1, &grad_refl);
    }
    coord = ClipIndexesWithMask(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_refl * grad_clip;
  }

  return coord;
}

template <typename T>
static __forceinline__ __device__ void
AtomicAdd3D(T *data, int64_t d, int64_t h, int64_t w, int64_t sD, int64_t sH,
            int64_t sW, int64_t D, int64_t H, int64_t W, T delta) {
  if (InBounds3D(d, h, w, D, H, W)) {
    atomicAdd(data + d * sD + h * sH + w * sW, delta);
  }
}

template <typename T, typename index_t>
__global__ void GridSample3DCudaBackwardKernel(
    const index_t nthreads, const T *grad_output, const T *input, const T *grid,
    index_t out_c, index_t out_d, index_t out_h, index_t out_w, index_t in_d,
    index_t in_h, index_t in_w, T *grad_input, T *grad_grid, const Mode mode,
    const PaddingMode padding_mode, bool align_corners) {
  index_t inp_sW = 1;
  index_t inp_sH = in_w;
  index_t inp_sD = in_h * in_w;
  index_t inp_sC = in_d * inp_sD;
  index_t inp_sN = out_c * inp_sC;

  index_t grid_sCoor = 1;
  index_t grid_sW = 3;
  index_t grid_sH = out_w * grid_sW;
  index_t grid_sD = out_h * grid_sH;
  index_t grid_sN = out_d * grid_sD;

  index_t gOut_sW = 1;
  index_t gOut_sH = out_w;
  index_t gOut_sD = out_h * out_w;
  index_t gOut_sC = out_d * gOut_sD;
  index_t gOut_sN = out_c * gOut_sC;

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_w;
    const index_t h = (index / out_w) % out_h;
    const index_t d = (index / (out_h * out_w)) % out_d;
    const index_t n = index / (out_d * out_h * out_w);
    const auto grid_offset =
        n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z co-ordinates from grid
    T ix = grid[grid_offset];
    T iy = grid[grid_offset + grid_sCoor];
    T iz = grid[grid_offset + 2 * grid_sCoor];

    // multipliers for gradients on ix, iy, and iz
    T gix_mult, giy_mult, giz_mult;
    ix = ComputePositionsWithMask(ix, in_w, padding_mode, align_corners,
                                  &gix_mult);
    iy = ComputePositionsWithMask(iy, in_h, padding_mode, align_corners,
                                  &giy_mult);
    iz = ComputePositionsWithMask(iz, in_d, padding_mode, align_corners,
                                  &giz_mult);

    if (mode == Mode::bilinear) {
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      index_t ix_tnw = static_cast<index_t>(std::floor(ix));
      index_t iy_tnw = static_cast<index_t>(std::floor(iy));
      index_t iz_tnw = static_cast<index_t>(std::floor(iz));

      index_t ix_tne = ix_tnw + 1;
      index_t iy_tne = iy_tnw;
      index_t iz_tne = iz_tnw;

      index_t ix_tsw = ix_tnw;
      index_t iy_tsw = iy_tnw + 1;
      index_t iz_tsw = iz_tnw;

      index_t ix_tse = ix_tnw + 1;
      index_t iy_tse = iy_tnw + 1;
      index_t iz_tse = iz_tnw;

      index_t ix_bnw = ix_tnw;
      index_t iy_bnw = iy_tnw;
      index_t iz_bnw = iz_tnw + 1;

      index_t ix_bne = ix_tnw + 1;
      index_t iy_bne = iy_tnw;
      index_t iz_bne = iz_tnw + 1;

      index_t ix_bsw = ix_tnw;
      index_t iy_bsw = iy_tnw + 1;
      index_t iz_bsw = iz_tnw + 1;

      index_t ix_bse = ix_tnw + 1;
      index_t iy_bse = iy_tnw + 1;
      index_t iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      T tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
      T tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
      T tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
      T tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
      T bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
      T bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
      T bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
      T bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

      T gix = static_cast<T>(0), giy = static_cast<T>(0),
        giz = static_cast<T>(0);
      index_t gOut_offset =
          n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
      index_t inp_offset_NC = n * inp_sN;
      T *gInp_ptr_NC = grad_input + n * inp_sN;
      for (index_t c = 0; c < out_c; ++c, gOut_offset += gOut_sC,
                   gInp_ptr_NC += inp_sC, inp_offset_NC += inp_sC) {
        T gOut = grad_output[gOut_offset];

        AtomicAdd3D(gInp_ptr_NC, iz_tnw, iy_tnw, ix_tnw, inp_sD, inp_sH, inp_sW,
                    in_d, in_h, in_w, tnw * gOut);
        AtomicAdd3D(gInp_ptr_NC, iz_tne, iy_tne, ix_tne, inp_sD, inp_sH, inp_sW,
                    in_d, in_h, in_w, tne * gOut);
        AtomicAdd3D(gInp_ptr_NC, iz_tsw, iy_tsw, ix_tsw, inp_sD, inp_sH, inp_sW,
                    in_d, in_h, in_w, tsw * gOut);
        AtomicAdd3D(gInp_ptr_NC, iz_tse, iy_tse, ix_tse, inp_sD, inp_sH, inp_sW,
                    in_d, in_h, in_w, tse * gOut);
        AtomicAdd3D(gInp_ptr_NC, iz_bnw, iy_bnw, ix_bnw, inp_sD, inp_sH, inp_sW,
                    in_d, in_h, in_w, bnw * gOut);
        AtomicAdd3D(gInp_ptr_NC, iz_bne, iy_bne, ix_bne, inp_sD, inp_sH, inp_sW,
                    in_d, in_h, in_w, bne * gOut);
        AtomicAdd3D(gInp_ptr_NC, iz_bsw, iy_bsw, ix_bsw, inp_sD, inp_sH, inp_sW,
                    in_d, in_h, in_w, bsw * gOut);
        AtomicAdd3D(gInp_ptr_NC, iz_bse, iy_bse, ix_bse, inp_sD, inp_sH, inp_sW,
                    in_d, in_h, in_w, bse * gOut);

        // calculate grad_grid
        if (InBounds3D(iz_tnw, iy_tnw, ix_tnw, in_d, in_h, in_w)) {
          T tnw_val = input[inp_offset_NC + iz_tnw * inp_sD + iy_tnw * inp_sH +
                            ix_tnw * inp_sW];
          gix -= tnw_val * (iy_bse - iy) * (iz_bse - iz) * gOut;
          giy -= tnw_val * (ix_bse - ix) * (iz_bse - iz) * gOut;
          giz -= tnw_val * (ix_bse - ix) * (iy_bse - iy) * gOut;
        }
        if (InBounds3D(iz_tne, iy_tne, ix_tne, in_d, in_h, in_w)) {
          T tne_val = input[inp_offset_NC + iz_tne * inp_sD + iy_tne * inp_sH +
                            ix_tne * inp_sW];
          gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gOut;
          giy -= tne_val * (ix - ix_bsw) * (iz_bsw - iz) * gOut;
          giz -= tne_val * (ix - ix_bsw) * (iy_bsw - iy) * gOut;
        }
        if (InBounds3D(iz_tsw, iy_tsw, ix_tsw, in_d, in_h, in_w)) {
          T tsw_val = input[inp_offset_NC + iz_tsw * inp_sD + iy_tsw * inp_sH +
                            ix_tsw * inp_sW];
          gix -= tsw_val * (iy - iy_bne) * (iz_bne - iz) * gOut;
          giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * gOut;
          giz -= tsw_val * (ix_bne - ix) * (iy - iy_bne) * gOut;
        }
        if (InBounds3D(iz_tse, iy_tse, ix_tse, in_d, in_h, in_w)) {
          T tse_val = input[inp_offset_NC + iz_tse * inp_sD + iy_tse * inp_sH +
                            ix_tse * inp_sW];
          gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gOut;
          giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * gOut;
          giz -= tse_val * (ix - ix_bnw) * (iy - iy_bnw) * gOut;
        }
        if (InBounds3D(iz_bnw, iy_bnw, ix_bnw, in_d, in_h, in_w)) {
          T bnw_val = input[inp_offset_NC + iz_bnw * inp_sD + iy_bnw * inp_sH +
                            ix_bnw * inp_sW];
          gix -= bnw_val * (iy_tse - iy) * (iz - iz_tse) * gOut;
          giy -= bnw_val * (ix_tse - ix) * (iz - iz_tse) * gOut;
          giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * gOut;
        }
        if (InBounds3D(iz_bne, iy_bne, ix_bne, in_d, in_h, in_w)) {
          T bne_val = input[inp_offset_NC + iz_bne * inp_sD + iy_bne * inp_sH +
                            ix_bne * inp_sW];
          gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gOut;
          giy -= bne_val * (ix - ix_tsw) * (iz - iz_tsw) * gOut;
          giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * gOut;
        }
        if (InBounds3D(iz_bsw, iy_bsw, ix_bsw, in_d, in_h, in_w)) {
          T bsw_val = input[inp_offset_NC + iz_bsw * inp_sD + iy_bsw * inp_sH +
                            ix_bsw * inp_sW];
          gix -= bsw_val * (iy - iy_tne) * (iz - iz_tne) * gOut;
          giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * gOut;
          giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * gOut;
        }
        if (InBounds3D(iz_bse, iy_bse, ix_bse, in_d, in_h, in_w)) {
          T bse_val = input[inp_offset_NC + iz_bse * inp_sD + iy_bse * inp_sH +
                            ix_bse * inp_sW];
          gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gOut;
          giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * gOut;
          giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * gOut;
        }
      }
      if (grad_grid != nullptr) {
        T *gGrid_ptr_NDHW = grad_grid + index * grid_sW;
        gGrid_ptr_NDHW[0] = gix_mult * gix;
        gGrid_ptr_NDHW[1] = giy_mult * giy;
        gGrid_ptr_NDHW[2] = giz_mult * giz;
      }
    } else if (mode == Mode::nearest) {
      auto ix_nearest = static_cast<index_t>(std::round(ix));
      auto iy_nearest = static_cast<index_t>(std::round(iy));
      auto iz_nearest = static_cast<index_t>(std::round(iz));

      // assign nearest neighbor pixel value to output pixel
      index_t gOut_offset =
          n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
      T *gInp_ptr_NC = grad_input + n * inp_sN;
      for (index_t c = 0; c < out_c;
           ++c, gOut_offset += gOut_sC, gInp_ptr_NC += inp_sC) {
        AtomicAdd3D(gInp_ptr_NC, iz_nearest, iy_nearest, ix_nearest, inp_sD,
                    inp_sH, inp_sW, in_d, in_h, in_w, grad_output[gOut_offset]);
      }
      if (grad_grid != nullptr) {
        T *gGrid_ptr_NDHW = grad_grid + index * grid_sW;
        gGrid_ptr_NDHW[0] = static_cast<T>(0);
        gGrid_ptr_NDHW[1] = static_cast<T>(0);
        gGrid_ptr_NDHW[2] = static_cast<T>(0);
      }
    }
  }
}

std::vector<paddle::Tensor>
GridSample3DCUDABackward(const paddle::Tensor &x, const paddle::Tensor &grid,
                         const paddle::Tensor &grad_out,
                         const std::string &mode,
                         const std::string &padding_mode, bool align_corners) {
  PaddingMode enum_padding_mode;
  Mode enum_mode;
  if (padding_mode == "border") {
    enum_padding_mode = PaddingMode::border;
  } else if (padding_mode == "reflection") {
    enum_padding_mode = PaddingMode::reflect;
  } else {
    enum_padding_mode = PaddingMode::zeros;
  }

  if (mode == "nearest") {
    enum_mode = Mode::nearest;
  } else {
    enum_mode = Mode::bilinear;
  }

  const int out_d = grid.shape()[1];
  const int out_h = grid.shape()[2];
  const int out_w = grid.shape()[3];
  const int n = x.shape()[0];
  const int c = x.shape()[1];
  const int in_d = x.shape()[2];
  const int in_h = x.shape()[3];
  const int in_w = x.shape()[4];

  auto grid_grad_output =
      paddle::empty({n, out_d, out_h, out_w, 3}, paddle::DataType::FLOAT32,
                    paddle::GPUPlace());
  auto x_grad_output =
      paddle::full({n, c, in_d, in_h, in_w}, 0, paddle::DataType::FLOAT32,
                   paddle::GPUPlace());

  const int count = static_cast<int>(n * out_d * out_h * out_w);

  int max_threads_per_block = 512;
  int block_num = (count - 1) / max_threads_per_block + 1;

  GridSample3DCudaBackwardKernel<float, int>
      <<<block_num, max_threads_per_block, 0, x.stream()>>>(
          count, grad_out.data<float>(), x.data<float>(), grid.data<float>(), c,
          out_d, out_h, out_w, in_d, in_h, in_w, x_grad_output.data<float>(),
          grid_grad_output.data<float>(), enum_mode, enum_padding_mode,
          align_corners);

  return {x_grad_output};
}

} // namespace paddle_custom_ops
} // namespace ultra_infer
