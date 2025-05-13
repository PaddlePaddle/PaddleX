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
/*! @brief Preprocessor object for table model.
 */
class ULTRAINFER_DECL StructureV2TablePreprocessor : public ProcessorManager {
public:
  StructureV2TablePreprocessor();
  using ProcessorManager::Run;
  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input data list, all the elements are FDMat
   * \param[in] outputs The output tensors which will be fed into runtime
   * \return true if the preprocess successed, otherwise false
   */
  bool Run(std::vector<FDMat> *images, std::vector<FDTensor> *outputs,
           size_t start_index, size_t end_index,
           const std::vector<int> &indices);

  /** \brief Implement the virtual function of ProcessorManager, Apply() is the
   *  body of Run(). Apply() contains the main logic of preprocessing, Run() is
   *  called by users to execute preprocessing
   *
   * \param[in] image_batch The input image batch
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess successed, otherwise false
   */
  virtual bool Apply(FDMatBatch *image_batch, std::vector<FDTensor> *outputs);

  /// Get the image info of the last batch, return a list of array
  /// {image width, image height, resize width, resize height}
  const std::vector<std::array<float, 6>> *GetBatchImgInfo() {
    return &batch_det_img_info_;
  }

private:
  void StructureV2TableResizeImage(FDMat *mat, int batch_idx);
  // for recording the switch of hwc2chw
  bool disable_permute_ = false;
  // for recording the switch of normalize
  bool disable_normalize_ = false;
  int max_len = 488;
  std::vector<int> rec_image_shape_ = {3, max_len, max_len};
  bool static_shape_infer_ = false;
  std::shared_ptr<Resize> resize_op_;
  std::shared_ptr<Pad> pad_op_;
  std::shared_ptr<Normalize> normalize_op_;
  std::shared_ptr<HWC2CHW> hwc2chw_op_;
  std::vector<std::array<float, 6>> batch_det_img_info_;
};

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
