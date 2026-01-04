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
#include "ultra_infer/vision/common/processors/manager.h"
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"

namespace ultra_infer {
namespace vision {

namespace ocr {
/*! @brief Preprocessor object for Classifier serials model.
 */
class ULTRAINFER_DECL ClassifierPreprocessor : public ProcessorManager {
public:
  ClassifierPreprocessor();
  using ProcessorManager::Run;
  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input data list, all the elements are FDMat
   * \param[in] outputs The output tensors which will be fed into runtime
   * \return true if the preprocess succeeded, otherwise false
   */
  bool Run(std::vector<FDMat> *images, std::vector<FDTensor> *outputs,
           size_t start_index, size_t end_index);

  /** \brief Implement the virtual function of ProcessorManager, Apply() is the
   *  body of Run(). Apply() contains the main logic of preprocessing, Run() is
   *  called by users to execute preprocessing
   *
   * \param[in] image_batch The input image batch
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess succeeded, otherwise false
   */
  virtual bool Apply(FDMatBatch *image_batch, std::vector<FDTensor> *outputs);

  /// Set preprocess normalize parameters, please call this API to customize
  /// the normalize parameters, otherwise it will use the default normalize
  /// parameters.
  void SetNormalize(const std::vector<float> &mean,
                    const std::vector<float> &std, bool is_scale) {
    normalize_op_ = std::make_shared<Normalize>(mean, std, is_scale);
  }

  /// Set cls_image_shape for the classification preprocess
  void SetClsImageShape(const std::vector<int> &cls_image_shape) {
    cls_image_shape_ = cls_image_shape;
  }
  /// Get cls_image_shape for the classification preprocess
  std::vector<int> GetClsImageShape() const { return cls_image_shape_; }

  /// This function will disable normalize in preprocessing step.
  void DisableNormalize() { disable_permute_ = true; }
  /// This function will disable hwc2chw in preprocessing step.
  void DisablePermute() { disable_normalize_ = true; }

private:
  void OcrClassifierResizeImage(FDMat *mat,
                                const std::vector<int> &cls_image_shape);
  // for recording the switch of hwc2chw
  bool disable_permute_ = false;
  // for recording the switch of normalize
  bool disable_normalize_ = false;
  std::vector<int> cls_image_shape_ = {3, 48, 192};

  std::shared_ptr<Resize> resize_op_;
  std::shared_ptr<Pad> pad_op_;
  std::shared_ptr<Normalize> normalize_op_;
  std::shared_ptr<HWC2CHW> hwc2chw_op_;
};

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
