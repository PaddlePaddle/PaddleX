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

namespace ultra_infer {

namespace vision {

namespace facedet {
/*! @brief UltraFace model object used when to load a UltraFace model exported
 * by UltraFace.
 */
class ULTRAINFER_DECL UltraFace : public UltraInferModel {
public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./ultraface.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in "valid_cpu_backends" \param[in] model_format
   * Model format of the loaded model, default is ONNX format
   */
  UltraFace(const std::string &model_file, const std::string &params_file = "",
            const RuntimeOption &custom_option = RuntimeOption(),
            const ModelFormat &model_format = ModelFormat::ONNX);

  std::string ModelName() const {
    return "Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB";
  }
  /** \brief Predict the face detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output face detection
   * result will be written to this structure \param[in] conf_threshold
   * confidence threshold for postprocessing, default is 0.7 \param[in]
   * nms_iou_threshold iou threshold for NMS, default is 0.3 \return true if
   * the prediction succeeded, otherwise false
   */
  virtual bool Predict(cv::Mat *im, FaceDetectionResult *result,
                       float conf_threshold = 0.7f,
                       float nms_iou_threshold = 0.3f);

  /*! @brief
  Argument for image preprocessing step, tuple of (width, height), decide the
  target size after resize, default (320, 240)
  */
  std::vector<int> size;

private:
  bool Initialize();

  bool Preprocess(Mat *mat, FDTensor *outputs,
                  std::map<std::string, std::array<float, 2>> *im_info);

  bool Postprocess(std::vector<FDTensor> &infer_result,
                   FaceDetectionResult *result,
                   const std::map<std::string, std::array<float, 2>> &im_info,
                   float conf_threshold, float nms_iou_threshold);

  bool IsDynamicInput() const { return is_dynamic_input_; }

  bool is_dynamic_input_;
};

} // namespace facedet
} // namespace vision
} // namespace ultra_infer
