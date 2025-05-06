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
/*! @brief Postprocessor object for DBDetector serials model.
 */

class ULTRAINFER_DECL StructureV2TablePostprocessor {
public:
  StructureV2TablePostprocessor();
  /** \brief Create a postprocessor instance for Recognizer serials model
   *
   * \param[in] label_path The path of label_dict
   * \param[in] box_shape Type of output box, default is ori.
   */

  explicit StructureV2TablePostprocessor(const std::string &dict_path,
                                         const std::string &box_shape);

  /** \brief Process the result of runtime and fill to RecognizerResult
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] texts The output text results of recognizer
   * \param[in] rec_scores The output score results of recognizer
   * \return true if the postprocess successed, otherwise false
   */
  bool Run(const std::vector<FDTensor> &tensors,
           std::vector<std::vector<std::array<int, 8>>> *bbox_batch_list,
           std::vector<std::vector<std::string>> *structure_batch_list,
           const std::vector<std::array<float, 6>> &batch_det_img_info_);

private:
  PostProcessor util_post_processor_;
  bool SingleBatchPostprocessor(
      const float *structure_probs, const float *bbox_preds, size_t slice_dim,
      size_t prob_dim, size_t box_dim, int img_width, int img_height,
      float ratio_h, float ratio_w, int pad_h, int pad_w,
      std::vector<std::array<int, 8>> *boxes_result,
      std::vector<std::string> *structure_list_result);

  bool merge_no_span_structure{true};
  std::vector<std::string> dict_character;
  std::string box_shape;
  std::vector<std::string> td_tokens{"<td>", "<td", "<td></td>"};
  std::map<std::string, int> dict;
  int ignore_beg_token_idx;
  int ignore_end_token_idx;
  int dict_end_idx;
  bool initialized_ = false;
};

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
