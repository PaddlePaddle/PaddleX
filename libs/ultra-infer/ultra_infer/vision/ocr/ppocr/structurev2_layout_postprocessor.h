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
namespace ocr {

/*! @brief Postprocessor object for PaddleDet serials model.
 */
class ULTRAINFER_DECL StructureV2LayoutPostprocessor {
public:
  StructureV2LayoutPostprocessor() {}
  /** \brief Process the result of runtime and fill to batch DetectionResult
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] results The output result of layout detection
   * \param[in] batch_layout_img_info The image info of input images,
   *            {{image width, image height, resize width, resize height},...}
   * \return true if the postprocess succeeded, otherwise false
   */
  bool Run(const std::vector<FDTensor> &tensors,
           std::vector<DetectionResult> *results,
           const std::vector<std::array<int, 4>> &batch_layout_img_info);

  /// Set score_threshold_ for layout detection postprocess, default is 0.4
  void SetScoreThreshold(float score_threshold) {
    score_threshold_ = score_threshold;
  }
  /// Set nms_threshold_ for layout detection postprocess, default is 0.5
  void SetNMSThreshold(float nms_threshold) { nms_threshold_ = nms_threshold; }
  /// Set num_class_ for layout detection postprocess, default is 5
  void SetNumClass(int num_class) { num_class_ = num_class; }
  /// Set fpn_stride_ for layout detection postprocess, default is {8, 16, 32,
  /// 64}
  void SetFPNStride(const std::vector<int> &fpn_stride) {
    fpn_stride_ = fpn_stride;
  }
  /// Set reg_max_ for layout detection postprocess, default is 8
  void SetRegMax(int reg_max) { reg_max_ = reg_max; } // should private ?
  /// Get score_threshold_ of layout detection postprocess, default is 0.4
  float GetScoreThreshold() const { return score_threshold_; }
  /// Get nms_threshold_ of layout detection postprocess, default is 0.5
  float GetNMSThreshold() const { return nms_threshold_; }
  /// Get num_class_ of layout detection postprocess, default is 5
  int GetNumClass() const { return num_class_; }
  /// Get fpn_stride_ of layout detection postprocess, default is {8, 16, 32,
  /// 64}
  std::vector<int> GetFPNStride() const { return fpn_stride_; }
  /// Get reg_max_ of layout detection postprocess, default is 8
  int GetRegMax() const { return reg_max_; }

private:
  std::array<float, 4> DisPred2Bbox(const std::vector<float> &bbox_pred, int x,
                                    int y, int stride, int resize_w,
                                    int resize_h, int reg_max);
  bool
  SingleBatchPostprocessor(const std::vector<FDTensor> &single_batch_tensors,
                           const std::array<int, 4> &layout_img_info,
                           DetectionResult *result);
  void SetSingleBatchExternalData(const std::vector<FDTensor> &tensors,
                                  std::vector<FDTensor> &single_batch_tensors,
                                  size_t batch_idx);

  std::vector<int> fpn_stride_ = {8, 16, 32, 64};
  float score_threshold_ = 0.4;
  float nms_threshold_ = 0.5;
  int num_class_ = 5;
  int reg_max_ = 8;
};

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
