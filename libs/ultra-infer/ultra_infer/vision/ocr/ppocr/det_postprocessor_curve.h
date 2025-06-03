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
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_postprocess_op.h"

namespace ultra_infer {
namespace vision {

namespace ocr {
/*! @brief Postprocessor object for DBCURVEDetector serials model.
 */
class ULTRAINFER_DECL DBCURVEDetectorPostprocessor {
public:
  /** \brief Process the result of runtime and fill to results structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] results The output result of detector
   * \param[in] batch_det_img_info The detector_preprocess result
   * \return true if the postprocess succeeded, otherwise false
   */
  bool Run(const std::vector<FDTensor> &tensors,
           std::vector<std::vector<std::vector<int>>> *results,
           const std::vector<std::array<int, 4>> &batch_det_img_info);

  /// Set det_db_thresh for the detection postprocess, default is 0.3
  void SetDetDBThresh(double det_db_thresh) { det_db_thresh_ = det_db_thresh; }
  /// Get det_db_thresh of the detection postprocess
  double GetDetDBThresh() const { return det_db_thresh_; }

  /// Set det_db_box_thresh for the detection postprocess, default is 0.6
  void SetDetDBBoxThresh(double det_db_box_thresh) {
    det_db_box_thresh_ = det_db_box_thresh;
  }
  /// Get det_db_box_thresh of the detection postprocess
  double GetDetDBBoxThresh() const { return det_db_box_thresh_; }

  /// Set det_db_unclip_ratio for the detection postprocess, default is 1.5
  void SetDetDBUnclipRatio(double det_db_unclip_ratio) {
    det_db_unclip_ratio_ = det_db_unclip_ratio;
  }
  /// Get det_db_unclip_ratio_ of the detection postprocess
  double GetDetDBUnclipRatio() const { return det_db_unclip_ratio_; }

  void SetDetDBScoreMode(const std::string &det_db_score_mode) {
    det_db_score_mode_ = det_db_score_mode;
  }

  void SetDetDBBoxType(const std::string &det_db_box_type) {
    det_db_box_type_ = det_db_box_type;
  }
  std::string GetDetDBScoreMode() const { return det_db_score_mode_; }

  std::string GetDetDBBoxType() const { return det_db_box_type_; }

  /// Set use_dilation for the detection postprocess, default is false
  void SetUseDilation(int use_dilation) { use_dilation_ = use_dilation; }
  /// Get use_dilation of the detection postprocess
  int GetUseDilation() const { return use_dilation_; }

private:
  double det_db_thresh_ = 0.3;
  double det_db_box_thresh_ = 0.6;
  double det_db_unclip_ratio_ = 1.5;
  std::string det_db_box_type_ = "bbox";
  std::string det_db_score_mode_ = "slow";
  bool use_dilation_ = false;
  PostProcessor util_post_processor_;
  bool SingleBatchPostprocessor(const float *out_data, int n2, int n3,
                                const std::array<int, 4> &det_img_info,
                                std::vector<std::vector<int>> *boxes_result);
};

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
