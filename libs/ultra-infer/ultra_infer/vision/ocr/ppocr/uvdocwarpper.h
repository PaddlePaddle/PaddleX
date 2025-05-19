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
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_postprocess_op.h"
#include "ultra_infer/vision/ocr/ppocr/uvdoc_postprocessor.h"
#include "ultra_infer/vision/ocr/ppocr/uvdoc_preprocessor.h"

namespace ultra_infer {
namespace vision {
/** \brief All OCR series model APIs are defined inside this namespace
 *
 */
namespace ocr {

/*! @brief UVDocWarpper object is used to load the detection model provided by
 * PaddleOCR.
 */
class ULTRAINFER_DECL UVDocWarpper : public UltraInferModel {
public:
  UVDocWarpper();
  /** \brief Set path of model file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g
   * ./ch_PP-OCRv3_det_infer/model.pdmodel. \param[in] params_file Path of
   * parameter file, e.g ./ch_PP-OCRv3_det_infer/model.pdiparams, if the model
   * format is ONNX, this parameter will be ignored. \param[in] custom_option
   * RuntimeOption for inference, the default will use cpu, and choose the
   * backend defined in `valid_cpu_backends`. \param[in] model_format Model
   * format of the loaded model, default is Paddle format.
   */
  UVDocWarpper(const std::string &model_file,
               const std::string &params_file = "",
               const RuntimeOption &custom_option = RuntimeOption(),
               const ModelFormat &model_format = ModelFormat::PADDLE);

  /** \brief Clone a new UVDocWarpper with less memory usage when multiple
   * instances of the same model are created
   *
   * \return new UVDocWarpper* type unique pointer
   */
  virtual std::unique_ptr<UVDocWarpper> Clone() const;

  /// Get model's name
  std::string ModelName() const { return "pp-uvdoc"; }

  /** \brief Predict the input image and get OCR detection model result.
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D
   * array with layout HWC, BGR format. \param[in] result The output of OCR
   * detection model result will be written to this structure. \return true if
   * the prediction is succeeded, otherwise false.
   */

  virtual bool Predict(cv::Mat *img, FDTensor *result);

  virtual bool Predict(const cv::Mat &img, FDTensor *result);

  /** \brief Predict the input image and get OCR detection model result.
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D
   * array with layout HWC, BGR format. \param[in] result The output of OCR
   * detection model result will be written to this structure. \return true if
   * the prediction is succeeded, otherwise false.
   */
  virtual bool BatchPredict(const std::vector<cv::Mat> &images,
                            std::vector<FDTensor> *results);

  /** \brief BatchPredict the input image and get OCR detection model result.
   *
   * \param[in] images The list input of image data, comes from cv::imread(), is
   * a 3-D array with layout HWC, BGR format. \param[in] results The output of
   * OCR detection model result will be written to this structure. \return true
   * if the prediction is succeeded, otherwise false.
   */
  /// Get preprocessor reference of UVDocWarpperPreprocessor
  virtual UVDocPreprocessor &GetPreprocessor() { return preprocessor_; }

  // / Get postprocessor reference of UVDocWarpperPostprocessor
  virtual UVDocPostprocessor &GetPostprocessor() { return postprocessor_; }

private:
  bool Initialize();
  UVDocPreprocessor preprocessor_;
  UVDocPostprocessor postprocessor_;
};

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
