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

#ifdef WITH_GPU

#include "adaptive_pool2d_kernel.h"

namespace ultra_infer {
template <typename T1, typename T2>
__global__ void CudaCastKernel(const T1 *in, T2 *out, int edge,
                               int out_bc_offset, int in_bc_offset, int ih,
                               int iw, int oh, int ow, bool is_avg) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= edge) {
    return;
  }
  int offset = floorf(float(position) / out_bc_offset);
  int h = floorf(float(position % out_bc_offset) / ow);
  int w = (position % out_bc_offset) % ow;
  int hstart = floorf(static_cast<float>(h * ih) / oh);
  int hend = ceilf(static_cast<float>((h + 1) * ih) / oh);
  int wstart = floorf(static_cast<float>(w * iw) / ow);
  int wend = ceilf(static_cast<float>((w + 1) * iw) / ow);
  float ele_val = 0.0;
  if (is_avg) {
    ele_val = 0.0;
  } else {
    ele_val =
        static_cast<float>(in[offset * in_bc_offset + hstart * iw + wstart]);
  }
  for (int h = hstart; h < hend; ++h) {
    for (int w = wstart; w < wend; ++w) {
      int input_idx = h * iw + w;
      if (is_avg) {
        ele_val =
            ele_val + static_cast<float>(in[offset * in_bc_offset + input_idx]);
      } else {
        ele_val =
            (ele_val >
             static_cast<float>(in[offset * in_bc_offset + input_idx]))
                ? ele_val
                : static_cast<float>(in[offset * in_bc_offset + input_idx]);
      }
    }
  }
  out[position] = static_cast<T2>(
      ele_val / static_cast<float>(((hend - hstart) * (wend - wstart))));
}

void CudaAdaptivePool(const std::vector<int64_t> &input_dims,
                      const std::vector<int64_t> &output_dims, void *output,
                      const void *input, void *compute_stream,
                      const std::string &pooling_type, const std::string &dtype,
                      const std::string &out_dtype) {
  auto casted_compute_stream = reinterpret_cast<cudaStream_t>(compute_stream);
  int out_bc_offset = output_dims[2] * output_dims[3];
  int in_bc_offset = input_dims[2] * input_dims[3];
  int jobs = 1;
  for (int i : output_dims) {
    jobs *= i;
  }
  bool is_avg = pooling_type == "avg";
  int threads = 256;
  int blocks = ceil(jobs / static_cast<float>(threads));
  if (dtype == "float") {
    CudaCastKernel<float, float><<<blocks, threads, 0, casted_compute_stream>>>(
        static_cast<const float *>(input), static_cast<float *>(output), jobs,
        out_bc_offset, in_bc_offset, int(input_dims[2]), int(input_dims[3]),
        int(output_dims[2]), int(output_dims[3]), is_avg);
  } else if (dtype == "half") {
    if (out_dtype == "half") {
      CudaCastKernel<half, half><<<blocks, threads, 0, casted_compute_stream>>>(
          static_cast<const half *>(input), static_cast<half *>(output), jobs,
          out_bc_offset, in_bc_offset, int(input_dims[2]), int(input_dims[3]),
          int(output_dims[2]), int(output_dims[3]), is_avg);
    }
    if (out_dtype == "float") {
      CudaCastKernel<half, float>
          <<<blocks, threads, 0, casted_compute_stream>>>(
              static_cast<const half *>(input), static_cast<float *>(output),
              jobs, out_bc_offset, in_bc_offset, int(input_dims[2]),
              int(input_dims[3]), int(output_dims[2]), int(output_dims[3]),
              is_avg);
    }
  }
}
} // namespace ultra_infer
#endif
