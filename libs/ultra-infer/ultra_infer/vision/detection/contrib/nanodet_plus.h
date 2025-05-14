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

namespace detection {
/*! @brief NanoDetPlus model object used when to load a NanoDetPlus model
 * exported by NanoDet.
 */
class ULTRAINFER_DECL NanoDetPlus : public UltraInferModel {
public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./nanodet_plus_320.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in "valid_cpu_backends" \param[in] model_format
   * Model format of the loaded model, default is ONNX format
   */
  NanoDetPlus(const std::string &model_file,
              const std::string &params_file = "",
              const RuntimeOption &custom_option = RuntimeOption(),
              const ModelFormat &model_format = ModelFormat::ONNX);
  /// Get model's name
  std::string ModelName() const { return "nanodet"; }

  /** \brief Predict the detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output detection result
   * will be written to this structure \param[in] conf_threshold confidence
   * threshold for postprocessing, default is 0.35 \param[in] nms_iou_threshold
   * iou threshold for NMS, default is 0.5 \return true if the prediction
   * successed, otherwise false
   */
  virtual bool Predict(cv::Mat *im, DetectionResult *result,
                       float conf_threshold = 0.35f,
                       float nms_iou_threshold = 0.5f);

  /*! @brief
  Argument for image preprocessing step, tuple of input size (width, height),
  default (320, 320)
  */
  std::vector<int> size;
  // padding value, size should be the same as channels
  std::vector<float> padding_value;
  // keep aspect ratio or not when perform resize operation.
  // This option is set as `false` by default in NanoDet-Plus
  bool keep_ratio;
  // downsample strides for NanoDet-Plus to generate anchors,
  // will take (8, 16, 32, 64) as default values
  std::vector<int> downsample_strides;
  // for offsetting the boxes by classes when using NMS, default 4096
  float max_wh;
  /*! @brief
  Argument for image postprocessing step, reg_max for GFL regression, default 7
  */
  int reg_max;

private:
  bool Initialize();

  bool Preprocess(Mat *mat, FDTensor *output,
                  std::map<std::string, std::array<float, 2>> *im_info);

  bool Postprocess(FDTensor &infer_result, DetectionResult *result,
                   const std::map<std::string, std::array<float, 2>> &im_info,
                   float conf_threshold, float nms_iou_threshold);

  bool IsDynamicInput() const { return is_dynamic_input_; }

  // whether to inference with dynamic shape (e.g ONNX export with dynamic shape
  // or not.)
  // RangiLyu/nanodet official 'export_onnx.py' script will export static ONNX
  // by default.
  // This value will auto check by ultra_infer after the internal Runtime
  // initialized.
  bool is_dynamic_input_;
};

} // namespace detection
} // namespace vision
} // namespace ultra_infer
