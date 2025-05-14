
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
/*! @brief YOLOR model object used when to load a YOLOR model exported by YOLOR.
 */
class ULTRAINFER_DECL YOLOR : public UltraInferModel {
public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./yolor.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in "valid_cpu_backends" \param[in] model_format
   * Model format of the loaded model, default is ONNX format
   */
  YOLOR(const std::string &model_file, const std::string &params_file = "",
        const RuntimeOption &custom_option = RuntimeOption(),
        const ModelFormat &model_format = ModelFormat::ONNX);

  virtual std::string ModelName() const { return "YOLOR"; }
  /** \brief Predict the detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread()
   * \param[in] result The output detection result will be written to this
   * structure \param[in] conf_threshold confidence threshold for
   * postprocessing, default is 0.25 \param[in] nms_iou_threshold iou threshold
   * for NMS, default is 0.5 \return true if the prediction successed, otherwise
   * false
   */
  virtual bool Predict(cv::Mat *im, DetectionResult *result,
                       float conf_threshold = 0.25,
                       float nms_iou_threshold = 0.5);

  /*! @brief
  Argument for image preprocessing step, tuple of (width, height), decide the
  target size after resize, default size = {640, 640}
  */
  std::vector<int> size;
  // padding value, size should be the same as channels

  std::vector<float> padding_value;
  // only pad to the minimum rectangle which height and width is times of stride
  bool is_mini_pad;
  // while is_mini_pad = false and is_no_pad = true,
  // will resize the image to the set size
  bool is_no_pad;
  // if is_scale_up is false, the input image only can be zoom out,
  // the maximum resize scale cannot exceed 1.0
  bool is_scale_up;
  // padding stride, for is_mini_pad
  int stride;
  // for offsetting the boxes by classes when using NMS
  float max_wh;

private:
  bool Initialize();

  bool Preprocess(Mat *mat, FDTensor *output,
                  std::map<std::string, std::array<float, 2>> *im_info);

  bool Postprocess(FDTensor &infer_result, DetectionResult *result,
                   const std::map<std::string, std::array<float, 2>> &im_info,
                   float conf_threshold, float nms_iou_threshold);

  void LetterBox(Mat *mat, const std::vector<int> &size,
                 const std::vector<float> &color, bool _auto,
                 bool scale_fill = false, bool scale_up = true,
                 int stride = 32);

  // whether to inference with dynamic shape (e.g ONNX export with dynamic shape
  // or not.)
  // while is_dynamic_shape if 'false', is_mini_pad will force 'false'. This
  // value will
  // auto check by ultra_infer after the internal Runtime already initialized.
  bool is_dynamic_input_;
};

} // namespace detection
} // namespace vision
} // namespace ultra_infer
