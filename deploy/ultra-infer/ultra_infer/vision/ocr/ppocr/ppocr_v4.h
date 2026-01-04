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

#include "ultra_infer/vision/ocr/ppocr/ppocr_v3.h"

namespace ultra_infer {
/** \brief This pipeline can launch detection model, classification model and
 * recognition model sequentially. All OCR pipeline APIs are defined inside this
 * namespace.
 *
 */
namespace pipeline {
/*! @brief PPOCRv4 is used to load PP-OCRv4 series models provided by PaddleOCR.
 */
class ULTRAINFER_DECL PPOCRv4 : public PPOCRv3 {
public:
  /** \brief Set up the detection model path, classification model path and
   * recognition model path respectively.
   *
   * \param[in] det_model Path of detection model, e.g ./ch_PP-OCRv4_det_infer
   * \param[in] cls_model Path of classification model, e.g
   * ./ch_ppocr_mobile_v2.0_cls_infer \param[in] rec_model Path of recognition
   * model, e.g ./ch_PP-OCRv4_rec_infer
   */
  PPOCRv4(ultra_infer::vision::ocr::DBDetector *det_model,
          ultra_infer::vision::ocr::Classifier *cls_model,
          ultra_infer::vision::ocr::Recognizer *rec_model)
      : PPOCRv3(det_model, cls_model, rec_model) {
    // The only difference between v2 and v3
    auto preprocess_shape = recognizer_->GetPreprocessor().GetRecImageShape();
    preprocess_shape[1] = 48;
    recognizer_->GetPreprocessor().SetRecImageShape(preprocess_shape);
  }
  /** \brief Classification model is optional, so this function is set up the
   * detection model path and recognition model path respectively.
   *
   * \param[in] det_model Path of detection model, e.g ./ch_PP-OCRv4_det_infer
   * \param[in] rec_model Path of recognition model, e.g ./ch_PP-OCRv4_rec_infer
   */
  PPOCRv4(ultra_infer::vision::ocr::DBDetector *det_model,
          ultra_infer::vision::ocr::Recognizer *rec_model)
      : PPOCRv3(det_model, rec_model) {
    // The only difference between v2 and v4
    auto preprocess_shape = recognizer_->GetPreprocessor().GetRecImageShape();
    preprocess_shape[1] = 48;
    recognizer_->GetPreprocessor().SetRecImageShape(preprocess_shape);
  }

  /** \brief Clone a new PPOCRv4 with less memory usage when multiple instances
   * of the same model are created
   *
   * \return new PPOCRv4* type unique pointer
   */
  std::unique_ptr<PPOCRv4> Clone() const {
    std::unique_ptr<PPOCRv4> clone_model =
        utils::make_unique<PPOCRv4>(PPOCRv4(*this));
    clone_model->detector_ = detector_->Clone().release();
    if (classifier_ != nullptr) {
      clone_model->classifier_ = classifier_->Clone().release();
    }
    clone_model->recognizer_ = recognizer_->Clone().release();
    return clone_model;
  }
};

} // namespace pipeline

namespace application {
namespace ocrsystem {
typedef pipeline::PPOCRv4 PPOCRSystemv4;
} // namespace ocrsystem
} // namespace application

} // namespace ultra_infer
