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

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#if defined(WITH_GPU)

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "iou3d_nms.h"

namespace ultra_infer {
namespace paddle_custom_ops {

#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")
// #define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
static inline int DIVUP(const int m, const int n) {
  return ((m) / (n) + ((m) % (n) > 0));
}

#define CHECK_ERROR(ans)                                                       \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#define D(x)                                                                   \
  PD_THROW('\n', x,                                                            \
           "\n--------------------------------- where is the error ? "         \
           "---------------------------------------\n");

static const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

void boxesoverlapLauncher(const int num_a, const float *boxes_a,
                          const int num_b, const float *boxes_b,
                          float *ans_overlap);
void boxesioubevLauncher(const int num_a, const float *boxes_a, const int num_b,
                         const float *boxes_b, float *ans_iou);
void nmsLauncher(const float *boxes, unsigned long long *mask, int boxes_num,
                 float nms_overlap_thresh);
void nmsNormalLauncher(const float *boxes, unsigned long long *mask,
                       int boxes_num, float nms_overlap_thresh);

int boxes_overlap_bev_gpu(paddle::Tensor boxes_a, paddle::Tensor boxes_b,
                          paddle::Tensor ans_overlap) {
  // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
  // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
  // params ans_overlap: (N, M)

  CHECK_INPUT(boxes_a);
  CHECK_INPUT(boxes_b);
  CHECK_INPUT(ans_overlap);

  int num_a = boxes_a.shape()[0];
  int num_b = boxes_b.shape()[0];

  const float *boxes_a_data = boxes_a.data<float>();
  const float *boxes_b_data = boxes_b.data<float>();
  float *ans_overlap_data = ans_overlap.data<float>();

  boxesoverlapLauncher(num_a, boxes_a_data, num_b, boxes_b_data,
                       ans_overlap_data);

  return 1;
}

int boxes_iou_bev_gpu(paddle::Tensor boxes_a, paddle::Tensor boxes_b,
                      paddle::Tensor ans_iou) {
  // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
  // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
  // params ans_overlap: (N, M)
  CHECK_INPUT(boxes_a);
  CHECK_INPUT(boxes_b);
  CHECK_INPUT(ans_iou);

  int num_a = boxes_a.shape()[0];
  int num_b = boxes_b.shape()[0];

  const float *boxes_a_data = boxes_a.data<float>();
  const float *boxes_b_data = boxes_b.data<float>();
  float *ans_iou_data = ans_iou.data<float>();

  boxesioubevLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_iou_data);

  return 1;
}

std::vector<paddle::Tensor> nms_gpu(const paddle::Tensor &boxes,
                                    float nms_overlap_thresh) {
  // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
  // params keep: (N)
  CHECK_INPUT(boxes);
  // CHECK_CONTIGUOUS(keep);
  auto keep = paddle::empty({boxes.shape()[0]}, paddle::DataType::INT32,
                            paddle::CPUPlace());
  auto num_to_keep_tensor =
      paddle::empty({1}, paddle::DataType::INT32, paddle::CPUPlace());
  int *num_to_keep_data = num_to_keep_tensor.data<int>();

  int boxes_num = boxes.shape()[0];
  const float *boxes_data = boxes.data<float>();
  int *keep_data = keep.data<int>();

  int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  unsigned long long *mask_data = NULL;
  CHECK_ERROR(cudaMalloc((void **)&mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long)));
  nmsLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

  // unsigned long long mask_cpu[boxes_num * col_blocks];
  // unsigned long long *mask_cpu = new unsigned long long [boxes_num *
  // col_blocks];
  std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

  //    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
  CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));

  cudaFree(mask_data);

  // WARN(qiuyanjun): codes below will throw a compile error on windows with
  // msvc. Thus, we chose to use std::vectored to store the result instead.
  // unsigned long long remv_cpu[col_blocks];
  // memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));
  std::vector<unsigned long long> remv_cpu(col_blocks, 0);

  int num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }

  num_to_keep_data[0] = num_to_keep;

  if (cudaSuccess != cudaGetLastError())
    printf("Error!\n");

  return {keep, num_to_keep_tensor};
}

int nms_normal_gpu(paddle::Tensor boxes, paddle::Tensor keep,
                   float nms_overlap_thresh) {
  // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
  // params keep: (N)

  CHECK_INPUT(boxes);
  // CHECK_CONTIGUOUS(keep);

  int boxes_num = boxes.shape()[0];
  const float *boxes_data = boxes.data<float>();
  // WARN(qiuyanjun): long type for Tensor::data() API is not exported by
  // paddle, it will raise some link error on windows with msvc. Please check:
  // https://github.com/PaddlePaddle/Paddle/blob/release/2.5/paddle/phi/api/lib/tensor.cc
#if defined(_WIN32)
  int *keep_data = keep.data<int>();
#else
  long *keep_data = keep.data<long>();
#endif

  int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  unsigned long long *mask_data = NULL;
  CHECK_ERROR(cudaMalloc((void **)&mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long)));
  nmsNormalLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

  // unsigned long long mask_cpu[boxes_num * col_blocks];
  // unsigned long long *mask_cpu = new unsigned long long [boxes_num *
  // col_blocks];
  std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

  //    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
  CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));

  cudaFree(mask_data);

  // WARN(qiuyanjun): codes below will throw a compile error on windows with
  // msvc. Thus, we chose to use std::vectored to store the result instead.
  // unsigned long long remv_cpu[col_blocks];
  // memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));
  std::vector<unsigned long long> remv_cpu(col_blocks, 0);

  int num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }
  if (cudaSuccess != cudaGetLastError())
    printf("Error!\n");

  return num_to_keep;
}

} // namespace paddle_custom_ops
} // namespace ultra_infer

#endif
