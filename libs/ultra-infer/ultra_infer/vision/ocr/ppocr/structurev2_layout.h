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
#include "ultra_infer/vision/ocr/ppocr/structurev2_layout_postprocessor.h"
#include "ultra_infer/vision/ocr/ppocr/structurev2_layout_preprocessor.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_postprocess_op.h"

namespace ultra_infer {
namespace vision {
namespace ocr {
/*! @brief StructureV2Layout object is used to load the PP-StructureV2-Layout
 * detection model.
 */
class ULTRAINFER_DECL StructureV2Layout : public UltraInferModel {
public:
  StructureV2Layout();
  /** \brief Set path of model file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g
   * ./picodet_lcnet_x1_0_fgd_layout_cdla_infer/model.pdmodel. \param[in]
   * params_file Path of parameter file, e.g
   * ./picodet_lcnet_x1_0_fgd_layout_cdla_infer/model.pdiparams, if the model
   * format is ONNX, this parameter will be ignored. \param[in] custom_option
   * RuntimeOption for inference, the default will use cpu, and choose the
   * backend defined in `valid_cpu_backends`. \param[in] model_format Model
   * format of the loaded model, default is Paddle format.
   */
  StructureV2Layout(const std::string &model_file,
                    const std::string &params_file = "",
                    const RuntimeOption &custom_option = RuntimeOption(),
                    const ModelFormat &model_format = ModelFormat::PADDLE);

  /** \brief Clone a new StructureV2Layout with less memory usage when multiple
   * instances of the same model are created
   *
   * \return newStructureV2Layout* type unique pointer
   */
  virtual std::unique_ptr<StructureV2Layout> Clone() const;

  /// Get model's name
  std::string ModelName() const { return "pp-structurev2-layout"; }

  /** \brief DEPRECATED Predict the detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output detection result
   * \return true if the prediction successed, otherwise false
   */
  virtual bool Predict(cv::Mat *im, DetectionResult *result);

  /** \brief Predict the detection result for an input image
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output detection result
   * \return true if the prediction successed, otherwise false
   */
  virtual bool Predict(const cv::Mat &im, DetectionResult *result);

  /** \brief Predict the detection result for an input image list
   * \param[in] im The input image list, all the elements come from
   * cv::imread(), is a 3-D array with layout HWC, BGR format \param[in] results
   * The output detection result list \return true if the prediction successed,
   * otherwise false
   */
  virtual bool BatchPredict(const std::vector<cv::Mat> &imgs,
                            std::vector<DetectionResult> *results);

  /// Get preprocessor reference ofStructureV2LayoutPreprocessor
  virtual StructureV2LayoutPreprocessor &GetPreprocessor() {
    return preprocessor_;
  }

  /// Get postprocessor reference ofStructureV2LayoutPostprocessor
  virtual StructureV2LayoutPostprocessor &GetPostprocessor() {
    return postprocessor_;
  }

private:
  bool Initialize();
  StructureV2LayoutPreprocessor preprocessor_;
  StructureV2LayoutPostprocessor postprocessor_;
};

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
