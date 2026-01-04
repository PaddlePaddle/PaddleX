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
//
// Part of the following code in this file refs to
// https://github.com/wang-xinyu/tensorrtx/blob/yolov5-v6.0/yolov5/preprocess.cu
//
// Copyright (c) 2022 tensorrtx
// Licensed under The MIT License
// \file preprocess.cu
// \brief
// \author Qi Liu, Xinyu Wang

#ifdef WITH_GPU
#include <opencv2/opencv.hpp>

#include "ultra_infer/vision/utils/cuda_utils.h"

namespace ultra_infer {
namespace vision {
namespace utils {

struct AffineMatrix {
  float value[6];
};

__global__ void
YoloPreprocessCudaKernel(uint8_t *src, int src_line_size, int src_width,
                         int src_height, float *dst, int dst_width,
                         int dst_height, uint8_t padding_color_b,
                         uint8_t padding_color_g, uint8_t padding_color_r,
                         AffineMatrix d2s, int edge) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= edge)
    return;

  float m_x1 = d2s.value[0];
  float m_y1 = d2s.value[1];
  float m_z1 = d2s.value[2];
  float m_x2 = d2s.value[3];
  float m_y2 = d2s.value[4];
  float m_z2 = d2s.value[5];

  int dx = position % dst_width;
  int dy = position / dst_width;
  float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
  float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
  float c0, c1, c2;

  if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
    // out of range
    c0 = padding_color_b;
    c1 = padding_color_g;
    c2 = padding_color_r;
  } else {
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    uint8_t const_value[] = {padding_color_b, padding_color_g, padding_color_r};
    float ly = src_y - y_low;
    float lx = src_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t *v1 = const_value;
    uint8_t *v2 = const_value;
    uint8_t *v3 = const_value;
    uint8_t *v4 = const_value;

    if (y_low >= 0) {
      if (x_low >= 0)
        v1 = src + y_low * src_line_size + x_low * 3;
      if (x_high < src_width)
        v2 = src + y_low * src_line_size + x_high * 3;
    }

    if (y_high < src_height) {
      if (x_low >= 0)
        v3 = src + y_high * src_line_size + x_low * 3;
      if (x_high < src_width)
        v4 = src + y_high * src_line_size + x_high * 3;
    }

    c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
    c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
    c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
  }

  // bgr to rgb
  float t = c2;
  c2 = c0;
  c0 = t;

  // normalization
  c0 = c0 / 255.0f;
  c1 = c1 / 255.0f;
  c2 = c2 / 255.0f;

  // rgbrgbrgb to rrrgggbbb
  int area = dst_width * dst_height;
  float *pdst_c0 = dst + dy * dst_width + dx;
  float *pdst_c1 = pdst_c0 + area;
  float *pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;
}

void CudaYoloPreprocess(uint8_t *src, int src_width, int src_height, float *dst,
                        int dst_width, int dst_height,
                        const std::vector<float> padding_value,
                        cudaStream_t stream) {
  AffineMatrix s2d, d2s;
  float scale =
      std::min(dst_height / (float)src_height, dst_width / (float)src_width);

  s2d.value[0] = scale;
  s2d.value[1] = 0;
  s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
  s2d.value[3] = 0;
  s2d.value[4] = scale;
  s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

  cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
  cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
  cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

  memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

  int jobs = dst_height * dst_width;
  int threads = 256;
  int blocks = ceil(jobs / (float)threads);
  YoloPreprocessCudaKernel<<<blocks, threads, 0, stream>>>(
      src, src_width * 3, src_width, src_height, dst, dst_width, dst_height,
      padding_value[0], padding_value[1], padding_value[2], d2s, jobs);
}

} // namespace utils
} // namespace vision
} // namespace ultra_infer
#endif
