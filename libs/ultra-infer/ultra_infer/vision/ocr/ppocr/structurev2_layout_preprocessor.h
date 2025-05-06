// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
/*! @brief Preprocessor object for DBDetector serials model.
 */
class ULTRAINFER_DECL StructureV2LayoutPreprocessor : public ProcessorManager {
public:
  StructureV2LayoutPreprocessor();

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] image_batch The input image batch
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess successed, otherwise false
   */
  virtual bool Apply(FDMatBatch *image_batch, std::vector<FDTensor> *outputs);

  /// Set preprocess normalize parameters, please call this API to customize
  /// the normalize parameters, otherwise it will use the default normalize
  /// parameters.
  void SetNormalize(const std::vector<float> &mean,
                    const std::vector<float> &std, bool is_scale) {
    normalize_permute_op_ =
        std::make_shared<NormalizeAndPermute>(mean, std, is_scale);
  }

  /// Get the image info of the last batch, return a list of array
  /// {image width, image height, resize width, resize height}
  const std::vector<std::array<int, 4>> *GetBatchLayoutImgInfo() {
    return &batch_layout_img_info_;
  }

  /// This function will disable normalize in preprocessing step.
  void DisableNormalize() { disable_permute_ = true; }
  /// This function will disable hwc2chw in preprocessing step.
  void DisablePermute() { disable_normalize_ = true; }
  /// Set image_shape for the detection preprocess.
  /// This api is usually used when you retrain the model.
  /// Generally, you do not need to use it.
  void SetLayoutImageShape(const std::vector<int> &image_shape) {
    layout_image_shape_ = image_shape;
  }
  /// Get cls_image_shape for the classification preprocess
  std::vector<int> GetLayoutImageShape() const { return layout_image_shape_; }
  /// Set static_shape_infer is true or not. When deploy PP-StructureV2
  /// on hardware which can not support dynamic input shape very well,
  /// like Huawei Ascned, static_shape_infer needs to to be true.
  void SetStaticShapeInfer(bool static_shape_infer) {
    static_shape_infer_ = static_shape_infer;
  }
  /// Get static_shape_infer of the recognition preprocess
  bool GetStaticShapeInfer() const { return static_shape_infer_; }

private:
  bool ResizeLayoutImage(FDMat *img, int resize_w, int resize_h);
  // for recording the switch of hwc2chw
  bool disable_permute_ = false;
  // for recording the switch of normalize
  bool disable_normalize_ = false;
  std::vector<std::array<int, 4>> batch_layout_img_info_;
  std::shared_ptr<Resize> resize_op_;
  std::shared_ptr<NormalizeAndPermute> normalize_permute_op_;
  std::vector<int> layout_image_shape_ = {3, 800, 608}; // c,h,w
  // default true for pp-structurev2-layout model, backbone picodet.
  bool static_shape_infer_ = true;
  std::array<int, 4> GetLayoutImgInfo(FDMat *img);
};

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
