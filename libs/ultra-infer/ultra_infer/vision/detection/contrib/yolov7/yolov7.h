// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.  //NOLINT
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
#include "ultra_infer/vision/detection/contrib/yolov7/postprocessor.h"
#include "ultra_infer/vision/detection/contrib/yolov7/preprocessor.h"

namespace ultra_infer {
namespace vision {
namespace detection {
/*! @brief YOLOv7 model object used when to load a YOLOv7 model exported by
 * YOLOv7.
 */
class ULTRAINFER_DECL YOLOv7 : public UltraInferModel {
public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./yolov7.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in "valid_cpu_backends" \param[in] model_format
   * Model format of the loaded model, default is ONNX format
   */
  YOLOv7(const std::string &model_file, const std::string &params_file = "",
         const RuntimeOption &custom_option = RuntimeOption(),
         const ModelFormat &model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "yolov7"; }

  /** \brief DEPRECATED Predict the detection result for an input image, remove
   * at 1.0 version
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output detection result
   * will be written to this structure \param[in] conf_threshold confidence
   * threshold for postprocessing, default is 0.25 \param[in] nms_threshold iou
   * threshold for NMS, default is 0.5 \return true if the prediction
   * succeeded, otherwise false
   */
  virtual bool Predict(cv::Mat *im, DetectionResult *result,
                       float conf_threshold = 0.25, float nms_threshold = 0.5);

  /** \brief Predict the detection result for an input image
   *
   * \param[in] img The input image data, comes from cv::imread(), is a 3-D
   * array with layout HWC, BGR format \param[in] result The output detection
   * result will be written to this structure \return true if the prediction
   * succeeded, otherwise false
   */
  virtual bool Predict(const cv::Mat &img, DetectionResult *result);

  /** \brief Predict the detection results for a batch of input images
   *
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output detection result list
   * \return true if the prediction succeeded, otherwise false
   */
  virtual bool BatchPredict(const std::vector<cv::Mat> &imgs,
                            std::vector<DetectionResult> *results);

  /// Get preprocessor reference of YOLOv7
  virtual YOLOv7Preprocessor &GetPreprocessor() { return preprocessor_; }

  /// Get postprocessor reference of YOLOv7
  virtual YOLOv7Postprocessor &GetPostprocessor() { return postprocessor_; }

protected:
  bool Initialize();
  YOLOv7Preprocessor preprocessor_;
  YOLOv7Postprocessor postprocessor_;
};

} // namespace detection
} // namespace vision
} // namespace ultra_infer
