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

#include "ultra_infer/vision/visualize/segmentation_arm.h"
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace ultra_infer {
namespace vision {

static constexpr int _OMP_THREADS = 2;

static inline void QuantizeBlendingWeight8(float weight,
                                           uint8_t *old_multi_factor,
                                           uint8_t *new_multi_factor) {
  // Quantize the weight to boost blending performance.
  // if 0.0 < w <= 1/8, w ~ 1/8=1/(2^3) shift right 3 mul 1, 7
  // if 1/8 < w <= 2/8, w ~ 2/8=1/(2^3) shift right 3 mul 2, 6
  // if 2/8 < w <= 3/8, w ~ 3/8=1/(2^3) shift right 3 mul 3, 5
  // if 3/8 < w <= 4/8, w ~ 4/8=1/(2^3) shift right 3 mul 4, 4
  // Shift factor is always 3, but the mul factor is different.
  // Moving 7 bits to the right tends to result in a zero value,
  // So, We choose to shift 3 bits to get an approximation.
  uint8_t weight_quantize = static_cast<uint8_t>(weight * 8.0f);
  *new_multi_factor = weight_quantize;
  *old_multi_factor = (8 - weight_quantize);
}

cv::Mat VisSegmentationNEON(const cv::Mat &im, const SegmentationResult &result,
                            float weight, bool quantize_weight) {
#ifndef __ARM_NEON
  FDASSERT(false, "UltraInfer was not compiled with Arm NEON support!")
#else
  int64_t height = result.shape[0];
  int64_t width = result.shape[1];
  auto vis_img = cv::Mat(height, width, CV_8UC3);

  int32_t size = static_cast<int32_t>(height * width);
  uint8_t *vis_ptr = static_cast<uint8_t *>(vis_img.data);
  const uint8_t *label_ptr =
      static_cast<const uint8_t *>(result.label_map.data());
  const uint8_t *im_ptr = static_cast<const uint8_t *>(im.data);

  if (!quantize_weight) {
    uint8x16_t zerox16 = vdupq_n_u8(0);
#pragma omp parallel for proc_bind(close) num_threads(_OMP_THREADS)
    for (int i = 0; i < size - 15; i += 16) {
      uint8x16x3_t bgrx16x3 = vld3q_u8(im_ptr + i * 3); // 48 bytes
      uint8x16_t labelx16 = vld1q_u8(label_ptr + i);    // 16 bytes
      uint8x16_t ibx16 = bgrx16x3.val[0];
      uint8x16_t igx16 = bgrx16x3.val[1];
      uint8x16_t irx16 = bgrx16x3.val[2];
      // e.g 0b00000001 << 7 -> 0b10000000 128;
      uint8x16_t mbx16 = vshlq_n_u8(labelx16, 7);
      uint8x16_t mgx16 = vshlq_n_u8(labelx16, 4);
      uint8x16_t mrx16 = vshlq_n_u8(labelx16, 3);
      uint8x16x3_t vbgrx16x3;
      // Keep the pixels of input im if mask = 0
      uint8x16_t cezx16 = vceqq_u8(labelx16, zerox16);
      vbgrx16x3.val[0] = vorrq_u8(vandq_u8(cezx16, ibx16), mbx16);
      vbgrx16x3.val[1] = vorrq_u8(vandq_u8(cezx16, igx16), mgx16);
      vbgrx16x3.val[2] = vorrq_u8(vandq_u8(cezx16, irx16), mrx16);
      vst3q_u8(vis_ptr + i * 3, vbgrx16x3);
    }
    for (int i = size - 15; i < size; i++) {
      uint8_t label = label_ptr[i];
      vis_ptr[i * 3 + 0] = (label << 7);
      vis_ptr[i * 3 + 1] = (label << 4);
      vis_ptr[i * 3 + 2] = (label << 3);
    }
    // Blend the colors use OpenCV
    cv::addWeighted(im, 1.0 - weight, vis_img, weight, 0, vis_img);
    return vis_img;
  }

  // Quantize the weight to boost blending performance.
  // After that, we can directly use shift instructions
  // to blend the colors from input im and mask. Please
  // check QuantizeBlendingWeight8 for more details.
  uint8_t old_multi_factor, new_multi_factor;
  QuantizeBlendingWeight8(weight, &old_multi_factor, &new_multi_factor);
  if (new_multi_factor == 0) {
    return im; // Only keep origin image.
  }

  if (new_multi_factor == 8) {
// Only keep mask, no need to blending with origin image.
#pragma omp parallel for proc_bind(close) num_threads(_OMP_THREADS)
    for (int i = 0; i < size - 15; i += 16) {
      uint8x16_t labelx16 = vld1q_u8(label_ptr + i); // 16 bytes
      // e.g 0b00000001 << 7 -> 0b10000000 128;
      uint8x16_t mbx16 = vshlq_n_u8(labelx16, 7);
      uint8x16_t mgx16 = vshlq_n_u8(labelx16, 4);
      uint8x16_t mrx16 = vshlq_n_u8(labelx16, 3);
      uint8x16x3_t vbgr16x3;
      vbgr16x3.val[0] = mbx16;
      vbgr16x3.val[1] = mgx16;
      vbgr16x3.val[2] = mrx16;
      vst3q_u8(vis_ptr + i * 3, vbgr16x3);
    }
    for (int i = size - 15; i < size; i++) {
      uint8_t label = label_ptr[i];
      vis_ptr[i * 3 + 0] = (label << 7);
      vis_ptr[i * 3 + 1] = (label << 4);
      vis_ptr[i * 3 + 2] = (label << 3);
    }
    return vis_img;
  }

  uint8x16_t zerox16 = vdupq_n_u8(0);
  uint8x16_t old_fx16 = vdupq_n_u8(old_multi_factor);
  uint8x16_t new_fx16 = vdupq_n_u8(new_multi_factor);
// Blend the two colors together with quantize 'weight'.
#pragma omp parallel for proc_bind(close) num_threads(_OMP_THREADS)
  for (int i = 0; i < size - 15; i += 16) {
    uint8x16x3_t bgrx16x3 = vld3q_u8(im_ptr + i * 3); // 48 bytes
    uint8x16_t labelx16 = vld1q_u8(label_ptr + i);    // 16 bytes
    uint8x16_t ibx16 = bgrx16x3.val[0];
    uint8x16_t igx16 = bgrx16x3.val[1];
    uint8x16_t irx16 = bgrx16x3.val[2];
    // e.g 0b00000001 << 7 -> 0b10000000 128;
    uint8x16_t mbx16 = vshlq_n_u8(labelx16, 7);
    uint8x16_t mgx16 = vshlq_n_u8(labelx16, 4);
    uint8x16_t mrx16 = vshlq_n_u8(labelx16, 3);
    // Moving 7 bits to the right tends to result in zero,
    // So, We choose to shift 3 bits to get an approximation
    uint8x16_t ibx16_mshr = vmulq_u8(vshrq_n_u8(ibx16, 3), old_fx16);
    uint8x16_t igx16_mshr = vmulq_u8(vshrq_n_u8(igx16, 3), old_fx16);
    uint8x16_t irx16_mshr = vmulq_u8(vshrq_n_u8(irx16, 3), old_fx16);
    uint8x16_t mbx16_mshr = vmulq_u8(vshrq_n_u8(mbx16, 3), new_fx16);
    uint8x16_t mgx16_mshr = vmulq_u8(vshrq_n_u8(mgx16, 3), new_fx16);
    uint8x16_t mrx16_mshr = vmulq_u8(vshrq_n_u8(mrx16, 3), new_fx16);
    uint8x16_t qbx16 = vqaddq_u8(ibx16_mshr, mbx16_mshr);
    uint8x16_t qgx16 = vqaddq_u8(igx16_mshr, mgx16_mshr);
    uint8x16_t qrx16 = vqaddq_u8(irx16_mshr, mrx16_mshr);
    // Keep the pixels of input im if label = 0 (means mask = 0)
    uint8x16_t cezx16 = vceqq_u8(labelx16, zerox16);
    uint8x16_t abx16 = vandq_u8(cezx16, ibx16);
    uint8x16_t agx16 = vandq_u8(cezx16, igx16);
    uint8x16_t arx16 = vandq_u8(cezx16, irx16);
    uint8x16x3_t vbgr16x3;
    // Reset qx values to 0 if label is 0, then, keep mask values
    // if label is not 0
    uint8x16_t ncezx16 = vmvnq_u8(cezx16);
    vbgr16x3.val[0] = vorrq_u8(abx16, vandq_u8(ncezx16, qbx16));
    vbgr16x3.val[1] = vorrq_u8(agx16, vandq_u8(ncezx16, qgx16));
    vbgr16x3.val[2] = vorrq_u8(arx16, vandq_u8(ncezx16, qrx16));
    // Store the blended pixels to vis img
    vst3q_u8(vis_ptr + i * 3, vbgr16x3);
  }
  for (int i = size - 15; i < size; i++) {
    uint8_t label = label_ptr[i];
    vis_ptr[i * 3 + 0] = (im_ptr[i * 3 + 0] >> 3) * old_multi_factor +
                         ((label << 7) >> 3) * new_multi_factor;
    vis_ptr[i * 3 + 1] = (im_ptr[i * 3 + 1] >> 3) * old_multi_factor +
                         ((label << 4) >> 3) * new_multi_factor;
    vis_ptr[i * 3 + 2] = (im_ptr[i * 3 + 2] >> 3) * old_multi_factor +
                         ((label << 3) >> 3) * new_multi_factor;
  }
  return vis_img;
#endif
}

} // namespace vision
} // namespace ultra_infer
