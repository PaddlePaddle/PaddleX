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

#pragma once
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"

namespace ultra_infer {
namespace vision {

namespace detection {
/*! @brief Preprocessor object for YOLOv5 serials model.
 */
class ULTRAINFER_DECL YOLOv5Preprocessor {
public:
  /** \brief Create a preprocessor instance for YOLOv5 serials model
   */
  YOLOv5Preprocessor();

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned
   * by cv::imread() \param[in] outputs The output tensors which will feed in
   * runtime \param[in] ims_info The shape info list, record input_shape and
   * output_shape \return true if the preprocess succeeded, otherwise false
   */
  bool Run(std::vector<FDMat> *images, std::vector<FDTensor> *outputs,
           std::vector<std::map<std::string, std::array<float, 2>>> *ims_info);

  /// Set target size, tuple of (width, height), default size = {640, 640}
  void SetSize(const std::vector<int> &size) { size_ = size; }

  /// Get target size, tuple of (width, height), default size = {640, 640}
  std::vector<int> GetSize() const { return size_; }

  /// Set padding value, size should be the same as channels
  void SetPaddingValue(const std::vector<float> &padding_value) {
    padding_value_ = padding_value;
  }

  /// Get padding value, size should be the same as channels
  std::vector<float> GetPaddingValue() const { return padding_value_; }

  /// Set is_scale_up, if is_scale_up is false, the input image only
  /// can be zoom out, the maximum resize scale cannot exceed 1.0, default true
  void SetScaleUp(bool is_scale_up) { is_scale_up_ = is_scale_up; }

  /// Get is_scale_up, default true
  bool GetScaleUp() const { return is_scale_up_; }

  /// Set is_mini_pad, pad to the minimum rectangle
  /// which height and width is times of stride
  void SetMiniPad(bool is_mini_pad) { is_mini_pad_ = is_mini_pad; }

  /// Get is_mini_pad, default false
  bool GetMiniPad() const { return is_mini_pad_; }

  /// Set padding stride, only for mini_pad mode
  void SetStride(int stride) { stride_ = stride; }

  /// Get padding stride, default 32
  bool GetStride() const { return stride_; }

protected:
  bool Preprocess(FDMat *mat, FDTensor *output,
                  std::map<std::string, std::array<float, 2>> *im_info);

  void LetterBox(FDMat *mat);

  // target size, tuple of (width, height), default size = {640, 640}
  std::vector<int> size_;

  // padding value, size should be the same as channels
  std::vector<float> padding_value_;

  // only pad to the minimum rectangle which height and width is times of stride
  bool is_mini_pad_;

  // while is_mini_pad = false and is_no_pad = true,
  // will resize the image to the set size
  bool is_no_pad_;

  // if is_scale_up is false, the input image only can be zoom out,
  // the maximum resize scale cannot exceed 1.0
  bool is_scale_up_;

  // padding stride, for is_mini_pad
  int stride_;

  // for offsetting the boxes by classes when using NMS
  float max_wh_;
};

} // namespace detection
} // namespace vision
} // namespace ultra_infer
