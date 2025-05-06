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

#include "ultra_infer/vision/visualize/swap_background_arm.h"

#include "ultra_infer/vision/visualize/visualize.h"
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "ultra_infer/utils/utils.h"

namespace ultra_infer {
namespace vision {

static constexpr int _OMP_THREADS = 2;

cv::Mat SwapBackgroundNEON(const cv::Mat &im, const cv::Mat &background,
                           const MattingResult &result,
                           bool remove_small_connected_area) {
#ifndef __ARM_NEON
  FDASSERT(false, "UltraInfer was not compiled with Arm NEON support!");
#else
  FDASSERT((!im.empty()), "Image can't be empty!");
  FDASSERT((im.channels() == 3), "Only support 3 channels image mat!");
  FDASSERT((!background.empty()), "Background image can't be empty!");
  FDASSERT((background.channels() == 3),
           "Only support 3 channels background image mat!");
  int out_h = static_cast<int>(result.shape[0]);
  int out_w = static_cast<int>(result.shape[1]);
  int height = im.rows;
  int width = im.cols;
  int bg_height = background.rows;
  int bg_width = background.cols;

  // WARN: may change the original alpha
  float *alpha_ptr = const_cast<float *>(result.alpha.data());

  cv::Mat alpha(out_h, out_w, CV_32FC1, alpha_ptr);
  if (remove_small_connected_area) {
    alpha = Visualize::RemoveSmallConnectedArea(alpha, 0.05f);
  }
  auto vis_img = cv::Mat(height, width, CV_8UC3);

  cv::Mat background_ref;
  if ((bg_height != height) || (bg_width != width)) {
    cv::resize(background, background_ref, cv::Size(width, height));
  } else {
    background_ref = background; // ref only
  }
  if ((background_ref).type() != CV_8UC3) {
    (background_ref).convertTo((background_ref), CV_8UC3);
  }

  if ((out_h != height) || (out_w != width)) {
    cv::resize(alpha, alpha, cv::Size(width, height));
  }

  uint8_t *vis_data = static_cast<uint8_t *>(vis_img.data);
  const uint8_t *background_data =
      static_cast<const uint8_t *>(background_ref.data);
  const uint8_t *im_data = static_cast<const uint8_t *>(im.data);
  const float *alpha_data = reinterpret_cast<const float *>(alpha.data);

  const int32_t size = static_cast<int32_t>(height * width);
#pragma omp parallel for proc_bind(close) num_threads(_OMP_THREADS)
  for (int i = 0; i < size - 7; i += 8) {
    uint8x8x3_t ibgrx8x3 = vld3_u8(im_data + i * 3); // 24 bytes
    // u8 -> u16 -> u32 -> f32
    uint16x8_t ibx8 = vmovl_u8(ibgrx8x3.val[0]);
    uint16x8_t igx8 = vmovl_u8(ibgrx8x3.val[1]);
    uint16x8_t irx8 = vmovl_u8(ibgrx8x3.val[2]);
    uint8x8x3_t bbgrx8x3 = vld3_u8(background_data + i * 3); // 24 bytes
    uint16x8_t bbx8 = vmovl_u8(bbgrx8x3.val[0]);
    uint16x8_t bgx8 = vmovl_u8(bbgrx8x3.val[1]);
    uint16x8_t brx8 = vmovl_u8(bbgrx8x3.val[2]);

    uint32x4_t hibx4 = vmovl_u16(vget_high_u16(ibx8));
    uint32x4_t higx4 = vmovl_u16(vget_high_u16(igx8));
    uint32x4_t hirx4 = vmovl_u16(vget_high_u16(irx8));
    uint32x4_t libx4 = vmovl_u16(vget_low_u16(ibx8));
    uint32x4_t ligx4 = vmovl_u16(vget_low_u16(igx8));
    uint32x4_t lirx4 = vmovl_u16(vget_low_u16(irx8));

    uint32x4_t hbbx4 = vmovl_u16(vget_high_u16(bbx8));
    uint32x4_t hbgx4 = vmovl_u16(vget_high_u16(bgx8));
    uint32x4_t hbrx4 = vmovl_u16(vget_high_u16(brx8));
    uint32x4_t lbbx4 = vmovl_u16(vget_low_u16(bbx8));
    uint32x4_t lbgx4 = vmovl_u16(vget_low_u16(bgx8));
    uint32x4_t lbrx4 = vmovl_u16(vget_low_u16(brx8));

    float32x4_t fhibx4 = vcvtq_f32_u32(hibx4);
    float32x4_t fhigx4 = vcvtq_f32_u32(higx4);
    float32x4_t fhirx4 = vcvtq_f32_u32(hirx4);
    float32x4_t flibx4 = vcvtq_f32_u32(libx4);
    float32x4_t fligx4 = vcvtq_f32_u32(ligx4);
    float32x4_t flirx4 = vcvtq_f32_u32(lirx4);

    float32x4_t fhbbx4 = vcvtq_f32_u32(hbbx4);
    float32x4_t fhbgx4 = vcvtq_f32_u32(hbgx4);
    float32x4_t fhbrx4 = vcvtq_f32_u32(hbrx4);
    float32x4_t flbbx4 = vcvtq_f32_u32(lbbx4);
    float32x4_t flbgx4 = vcvtq_f32_u32(lbgx4);
    float32x4_t flbrx4 = vcvtq_f32_u32(lbrx4);

    // alpha load from little end
    float32x4_t lalpx4 = vld1q_f32(alpha_data + i);     // low bits
    float32x4_t halpx4 = vld1q_f32(alpha_data + i + 4); // high bits
    float32x4_t rlalpx4 = vsubq_f32(vdupq_n_f32(1.0f), lalpx4);
    float32x4_t rhalpx4 = vsubq_f32(vdupq_n_f32(1.0f), halpx4);

    // blending
    float32x4_t fhvbx4 =
        vaddq_f32(vmulq_f32(fhibx4, halpx4), vmulq_f32(fhbbx4, rhalpx4));
    float32x4_t fhvgx4 =
        vaddq_f32(vmulq_f32(fhigx4, halpx4), vmulq_f32(fhbgx4, rhalpx4));
    float32x4_t fhvrx4 =
        vaddq_f32(vmulq_f32(fhirx4, halpx4), vmulq_f32(fhbrx4, rhalpx4));
    float32x4_t flvbx4 =
        vaddq_f32(vmulq_f32(flibx4, lalpx4), vmulq_f32(flbbx4, rlalpx4));
    float32x4_t flvgx4 =
        vaddq_f32(vmulq_f32(fligx4, lalpx4), vmulq_f32(flbgx4, rlalpx4));
    float32x4_t flvrx4 =
        vaddq_f32(vmulq_f32(flirx4, lalpx4), vmulq_f32(flbrx4, rlalpx4));

    // f32 -> u32 -> u16 -> u8
    uint8x8x3_t vbgrx8x3;
    // combine low 64 bits and high 64 bits into one 128 neon register
    vbgrx8x3.val[0] = vmovn_u16(vcombine_u16(vmovn_u32(vcvtq_u32_f32(flvbx4)),
                                             vmovn_u32(vcvtq_u32_f32(fhvbx4))));
    vbgrx8x3.val[1] = vmovn_u16(vcombine_u16(vmovn_u32(vcvtq_u32_f32(flvgx4)),
                                             vmovn_u32(vcvtq_u32_f32(fhvgx4))));
    vbgrx8x3.val[2] = vmovn_u16(vcombine_u16(vmovn_u32(vcvtq_u32_f32(flvrx4)),
                                             vmovn_u32(vcvtq_u32_f32(fhvrx4))));
    vst3_u8(vis_data + i * 3, vbgrx8x3);
  }

  for (int i = size - 7; i < size; i++) {
    float alp = alpha_data[i];
    for (int c = 0; c < 3; ++c) {
      vis_data[i * 3 + 0] = cv::saturate_cast<uchar>(
          static_cast<float>(im_data[i * 3 + c]) * alp +
          (1.0f - alp) * static_cast<float>(background_data[i * 3 + c]));
    }
  }

  return vis_img;
#endif
}

cv::Mat SwapBackgroundNEON(const cv::Mat &im, const cv::Mat &background,
                           const SegmentationResult &result,
                           int background_label) {
#ifndef __ARM_NEON
  FDASSERT(false, "UltraInfer was not compiled with Arm NEON support!")
#else
  FDASSERT((!im.empty()), "Image can't be empty!");
  FDASSERT((im.channels() == 3), "Only support 3 channels image mat!");
  FDASSERT((!background.empty()), "Background image can't be empty!");
  FDASSERT((background.channels() == 3),
           "Only support 3 channels background image mat!");
  int out_h = static_cast<int>(result.shape[0]);
  int out_w = static_cast<int>(result.shape[1]);
  int height = im.rows;
  int width = im.cols;
  int bg_height = background.rows;
  int bg_width = background.cols;
  auto vis_img = cv::Mat(height, width, CV_8UC3);

  cv::Mat background_ref;
  if ((bg_height != height) || (bg_width != width)) {
    cv::resize(background, background_ref, cv::Size(width, height));
  } else {
    background_ref = background; // ref only
  }
  if ((background_ref).type() != CV_8UC3) {
    (background_ref).convertTo((background_ref), CV_8UC3);
  }

  uint8_t *vis_data = static_cast<uint8_t *>(vis_img.data);
  const uint8_t *background_data =
      static_cast<const uint8_t *>(background_ref.data);
  const uint8_t *im_data = static_cast<const uint8_t *>(im.data);
  const uint8_t *label_data =
      static_cast<const uint8_t *>(result.label_map.data());

  const uint8_t background_label_ = static_cast<uint8_t>(background_label);
  const int32_t size = static_cast<int32_t>(height * width);

  uint8x16_t backgroundx16 = vdupq_n_u8(background_label_);
#pragma omp parallel for proc_bind(close) num_threads(_OMP_THREADS)
  for (int i = 0; i < size - 15; i += 16) {
    uint8x16x3_t ibgr16x3 = vld3q_u8(im_data + i * 3); // 48 bytes
    uint8x16x3_t bbgr16x3 = vld3q_u8(background_data + i * 3);
    uint8x16_t labelx16 = vld1q_u8(label_data + i); // 16 bytes
    // Set mask bit = 1 if label != background_label
    uint8x16_t nkeepx16 = vceqq_u8(labelx16, backgroundx16);
    uint8x16_t keepx16 = vmvnq_u8(nkeepx16); // keep_value = 1
    uint8x16x3_t vbgr16x3;
    vbgr16x3.val[0] = vorrq_u8(vandq_u8(ibgr16x3.val[0], keepx16),
                               vandq_u8(bbgr16x3.val[0], nkeepx16));
    vbgr16x3.val[1] = vorrq_u8(vandq_u8(ibgr16x3.val[1], keepx16),
                               vandq_u8(bbgr16x3.val[1], nkeepx16));
    vbgr16x3.val[2] = vorrq_u8(vandq_u8(ibgr16x3.val[2], keepx16),
                               vandq_u8(bbgr16x3.val[2], nkeepx16));
    // Store the blended pixels to vis img
    vst3q_u8(vis_data + i * 3, vbgr16x3);
  }

  for (int i = size - 15; i < size; i++) {
    uint8_t label = label_data[i];
    if (label != background_label_) {
      vis_data[i * 3 + 0] = im_data[i * 3 + 0];
      vis_data[i * 3 + 1] = im_data[i * 3 + 1];
      vis_data[i * 3 + 2] = im_data[i * 3 + 2];
    } else {
      vis_data[i * 3 + 0] = background_data[i * 3 + 0];
      vis_data[i * 3 + 1] = background_data[i * 3 + 1];
      vis_data[i * 3 + 2] = background_data[i * 3 + 2];
    }
  }

  return vis_img;
#endif
}

} // namespace vision
} // namespace ultra_infer
