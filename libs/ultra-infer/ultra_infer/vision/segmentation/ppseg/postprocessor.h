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
#include "ultra_infer/ultra_infer_model.h"
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"
#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace segmentation {
/*! @brief Postprocessor object for PaddleSeg serials model.
 */
class ULTRAINFER_DECL PaddleSegPostprocessor {
public:
  /** \brief Create a postprocessor instance for PaddleSeg serials model
   *
   * \param[in] config_file Path of configuration file for deployment, e.g
   * ppliteseg/deploy.yaml
   */
  explicit PaddleSegPostprocessor(const std::string &config_file);

  /** \brief Process the result of runtime and fill to SegmentationResult
   * structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of detection
   * \param[in] imgs_info The original input images shape info map, key is
   * "shape_info", value is vector<array<int, 2>> a{{height, width}} \return
   * true if the postprocess successed, otherwise false
   */
  virtual bool
  Run(const std::vector<FDTensor> &infer_results,
      std::vector<SegmentationResult> *results,
      const std::map<std::string, std::vector<std::array<int, 2>>> &imgs_info);

  /** \brief Get apply_softmax property of PaddleSeg model, default is false
   */
  bool GetApplySoftmax() const { return apply_softmax_; }

  /// Set apply_softmax value, bool type required
  void SetApplySoftmax(bool value) { apply_softmax_ = value; }

  /// Get store_score_map property of PaddleSeg model, default is false
  bool GetStoreScoreMap() const { return store_score_map_; }

  /// Set store_score_map value, bool type required
  void SetStoreScoreMap(bool value) { store_score_map_ = value; }

private:
  virtual bool ReadFromConfig(const std::string &config_file);

  virtual bool SliceOneResultFromBatchInferResults(
      const FDTensor &infer_results, FDTensor *infer_result,
      const std::vector<int64_t> &infer_result_shape, const int64_t &start_idx);

  virtual bool ProcessWithScoreResult(const FDTensor &infer_result,
                                      const int64_t &out_num,
                                      SegmentationResult *result);

  virtual bool ProcessWithLabelResult(const FDTensor &infer_result,
                                      const int64_t &out_num,
                                      SegmentationResult *result);

  bool is_with_softmax_ = false;

  bool is_with_argmax_ = true;

  bool apply_softmax_ = false;

  bool store_score_map_ = false;

  bool initialized_ = false;
};

} // namespace segmentation
} // namespace vision
} // namespace ultra_infer
