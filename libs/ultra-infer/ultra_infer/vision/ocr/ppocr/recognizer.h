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
#include "ultra_infer/vision/ocr/ppocr/rec_postprocessor.h"
#include "ultra_infer/vision/ocr/ppocr/rec_preprocessor.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_postprocess_op.h"

namespace ultra_infer {
namespace vision {
/** \brief All OCR series model APIs are defined inside this namespace
 *
 */
namespace ocr {
/*! @brief Recognizer object is used to load the recognition model provided by
 * PaddleOCR.
 */
class ULTRAINFER_DECL Recognizer : public UltraInferModel {
public:
  Recognizer();
  /** \brief Set path of model file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g
   * ./ch_PP-OCRv3_rec_infer/model.pdmodel. \param[in] params_file Path of
   * parameter file, e.g ./ch_PP-OCRv3_rec_infer/model.pdiparams, if the model
   * format is ONNX, this parameter will be ignored. \param[in] label_path Path
   * of label file used by OCR recognition model. e.g ./ppocr_keys_v1.txt
   * \param[in] custom_option RuntimeOption for inference, the default will use
   * cpu, and choose the backend defined in `valid_cpu_backends`. \param[in]
   * model_format Model format of the loaded model, default is Paddle format.
   */
  Recognizer(const std::string &model_file, const std::string &params_file = "",
             const std::string &label_path = "",
             const RuntimeOption &custom_option = RuntimeOption(),
             const ModelFormat &model_format = ModelFormat::PADDLE);

  /// Get model's name
  std::string ModelName() const { return "ppocr/ocr_rec"; }

  /** \brief Clone a new Recognizer with less memory usage when multiple
   * instances of the same model are created
   *
   * \return new Recognizer* type unique pointer
   */
  virtual std::unique_ptr<Recognizer> Clone() const;

  /** \brief Predict the input image and get OCR recognition model result.
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D
   * array with layout HWC, BGR format. \param[in] text The text result of rec
   * model will be written into this parameter. \param[in] rec_score The sccore
   * result of rec model will be written into this parameter. \return true if
   * the prediction is successed, otherwise false.
   */
  virtual bool Predict(const cv::Mat &img, std::string *text, float *rec_score);

  /** \brief Predict the input image and get OCR recognition model result.
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D
   * array with layout HWC, BGR format. \param[in] ocr_result The output of OCR
   * recognition model result will be written to this structure. \return true if
   * the prediction is successed, otherwise false.
   */
  virtual bool Predict(const cv::Mat &img, vision::OCRResult *ocr_result);

  /** \brief BatchPredict the input image and get OCR recognition model result.
   *
   * \param[in] images The list of input image data, comes from cv::imread(), is
   * a 3-D array with layout HWC, BGR format. \param[in] ocr_result The output
   * of OCR recognition model result will be written to this structure. \return
   * true if the prediction is successed, otherwise false.
   */
  virtual bool BatchPredict(const std::vector<cv::Mat> &images,
                            vision::OCRResult *ocr_result);

  /** \brief BatchPredict the input image and get OCR recognition model result.
   *
   * \param[in] images The list of input image data, comes from cv::imread(), is
   * a 3-D array with layout HWC, BGR format. \param[in] texts The list of text
   * results of rec model will be written into this vector. \param[in]
   * rec_scores The list of sccore result of rec model will be written into this
   * vector. \return true if the prediction is successed, otherwise false.
   */
  virtual bool BatchPredict(const std::vector<cv::Mat> &images,
                            std::vector<std::string> *texts,
                            std::vector<float> *rec_scores);

  virtual bool BatchPredict(const std::vector<cv::Mat> &images,
                            std::vector<std::string> *texts,
                            std::vector<float> *rec_scores, size_t start_index,
                            size_t end_index, const std::vector<int> &indices);

  /// Get preprocessor reference of DBDetectorPreprocessor
  virtual RecognizerPreprocessor &GetPreprocessor() { return preprocessor_; }

  /// Get postprocessor reference of DBDetectorPostprocessor
  virtual RecognizerPostprocessor &GetPostprocessor() { return postprocessor_; }

private:
  bool Initialize();
  RecognizerPreprocessor preprocessor_;
  RecognizerPostprocessor postprocessor_;
};

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
