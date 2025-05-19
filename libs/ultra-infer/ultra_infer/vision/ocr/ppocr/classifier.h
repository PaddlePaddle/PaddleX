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
#include "ultra_infer/utils/unique_ptr.h"
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"
#include "ultra_infer/vision/ocr/ppocr/cls_postprocessor.h"
#include "ultra_infer/vision/ocr/ppocr/cls_preprocessor.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_postprocess_op.h"

namespace ultra_infer {
namespace vision {
/** \brief All OCR series model APIs are defined inside this namespace
 *
 */
namespace ocr {
/*! @brief Classifier object is used to load the classification model provided
 * by PaddleOCR.
 */
class ULTRAINFER_DECL Classifier : public UltraInferModel {
public:
  Classifier();
  /** \brief Set path of model file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g
   * ./ch_ppocr_mobile_v2.0_cls_infer/model.pdmodel. \param[in] params_file Path
   * of parameter file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdiparams, if
   * the model format is ONNX, this parameter will be ignored. \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in `valid_cpu_backends`. \param[in] model_format
   * Model format of the loaded model, default is Paddle format.
   */
  Classifier(const std::string &model_file, const std::string &params_file = "",
             const RuntimeOption &custom_option = RuntimeOption(),
             const ModelFormat &model_format = ModelFormat::PADDLE);

  /** \brief Clone a new Classifier with less memory usage when multiple
   * instances of the same model are created
   *
   * \return new Classifier* type unique pointer
   */
  virtual std::unique_ptr<Classifier> Clone() const;

  /// Get model's name
  std::string ModelName() const { return "ppocr/ocr_cls"; }

  /** \brief Predict the input image and get OCR classification model
   * cls_result.
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D
   * array with layout HWC, BGR format. \param[in] cls_label The label result of
   * cls model will be written in to this param. \param[in] cls_score The score
   * result of cls model will be written in to this param. \return true if the
   * prediction is succeeded, otherwise false.
   */
  virtual bool Predict(const cv::Mat &img, int32_t *cls_label,
                       float *cls_score);

  /** \brief Predict the input image and get OCR recognition model result.
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D
   * array with layout HWC, BGR format. \param[in] ocr_result The output of OCR
   * recognition model result will be written to this structure. \return true if
   * the prediction is succeeded, otherwise false.
   */
  virtual bool Predict(const cv::Mat &img, vision::OCRResult *ocr_result);

  /** \brief BatchPredict the input image and get OCR classification model
   * result.
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D
   * array with layout HWC, BGR format. \param[in] ocr_result The output of OCR
   * classification model result will be written to this structure. \return true
   * if the prediction is succeeded, otherwise false.
   */
  virtual bool BatchPredict(const std::vector<cv::Mat> &images,
                            vision::OCRResult *ocr_result);

  /** \brief BatchPredict the input image and get OCR classification model
   * cls_result.
   *
   * \param[in] images The list of input image data, comes from cv::imread(), is
   * a 3-D array with layout HWC, BGR format. \param[in] cls_labels The label
   * results of cls model will be written in to this vector. \param[in]
   * cls_scores The score results of cls model will be written in to this
   * vector. \return true if the prediction is succeeded, otherwise false.
   */
  virtual bool BatchPredict(const std::vector<cv::Mat> &images,
                            std::vector<int32_t> *cls_labels,
                            std::vector<float> *cls_scores);
  virtual bool BatchPredict(const std::vector<cv::Mat> &images,
                            std::vector<int32_t> *cls_labels,
                            std::vector<float> *cls_scores, size_t start_index,
                            size_t end_index);

  /// Get preprocessor reference of ClassifierPreprocessor
  virtual ClassifierPreprocessor &GetPreprocessor() { return preprocessor_; }

  /// Get postprocessor reference of ClassifierPostprocessor
  virtual ClassifierPostprocessor &GetPostprocessor() { return postprocessor_; }

private:
  bool Initialize();
  ClassifierPreprocessor preprocessor_;
  ClassifierPostprocessor postprocessor_;
};

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
