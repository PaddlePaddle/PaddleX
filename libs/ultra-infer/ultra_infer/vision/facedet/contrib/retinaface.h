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
/** \brief All object face detection model APIs are defined inside this
 * namespace
 *
 */
namespace facedet {
/*! @brief RetinaFace model object used when to load a RetinaFace model exported
 * by RetinaFace.
 */
class ULTRAINFER_DECL RetinaFace : public UltraInferModel {
public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./retinaface.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in "valid_cpu_backends" \param[in] model_format
   * Model format of the loaded model, default is ONNX format
   */
  RetinaFace(const std::string &model_file, const std::string &params_file = "",
             const RuntimeOption &custom_option = RuntimeOption(),
             const ModelFormat &model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "Pytorch_Retinaface"; }
  /** \brief Predict the face detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output face detection
   * result will be written to this structure \param[in] conf_threshold
   * confidence threshold for postprocessing, default is 0.25 \param[in]
   * nms_iou_threshold iou threashold for NMS, default is 0.4 \return true if
   * the prediction successed, otherwise false
   */
  virtual bool Predict(cv::Mat *im, FaceDetectionResult *result,
                       float conf_threshold = 0.25f,
                       float nms_iou_threshold = 0.4f);

  /*! @brief
  Argument for image preprocessing step, tuple of (width, height), decide the
  target size after resize, default (640, 640)
  */
  std::vector<int> size;
  /*! @brief
  Argument for image postprocessing step, variance in RetinaFace's
  prior-box(anchor) generate process, default (0.1, 0.2)
  */
  std::vector<float> variance;
  /*! @brief
  Argument for image postprocessing step, downsample strides (namely, steps) for
  RetinaFace to generate anchors, will take (8,16,32) as default values
  */
  std::vector<int> downsample_strides;
  /*! @brief
  Argument for image postprocessing step, min sizes, width and height for each
  anchor, default min_sizes = {{16, 32}, {64, 128}, {256, 512}}
  */
  std::vector<std::vector<int>> min_sizes;
  /*! @brief
  Argument for image postprocessing step, landmarks_per_face, default 5 in
  RetinaFace
  */
  int landmarks_per_face;

private:
  bool Initialize();

  bool Preprocess(Mat *mat, FDTensor *output,
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
