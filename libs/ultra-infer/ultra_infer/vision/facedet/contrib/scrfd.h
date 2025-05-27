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
#include <unordered_map>

namespace ultra_infer {

namespace vision {

namespace facedet {
/*! @brief SCRFD model object used when to load a SCRFD model exported by SCRFD.
 */
class ULTRAINFER_DECL SCRFD : public UltraInferModel {
public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./scrfd.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in "valid_cpu_backends" \param[in] model_format
   * Model format of the loaded model, default is ONNX format
   */
  SCRFD(const std::string &model_file, const std::string &params_file = "",
        const RuntimeOption &custom_option = RuntimeOption(),
        const ModelFormat &model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "scrfd"; }
  /** \brief Predict the face detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output face detection
   * result will be written to this structure \param[in] conf_threshold
   * confidence threshold for postprocessing, default is 0.25 \param[in]
   * nms_iou_threshold iou threshold for NMS, default is 0.4 \return true if
   * the prediction succeeded, otherwise false
   */
  virtual bool Predict(cv::Mat *im, FaceDetectionResult *result,
                       float conf_threshold = 0.25f,
                       float nms_iou_threshold = 0.4f);

  /*! @brief
  Argument for image preprocessing step, tuple of (width, height), decide the
  target size after resize, default (640, 640)
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
  /*! @brief
  Argument for image postprocessing step, downsample strides (namely, steps) for
  SCRFD to generate anchors, will take (8,16,32) as default values
  */
  std::vector<int> downsample_strides;
  /*! @brief
  Argument for image postprocessing step, landmarks_per_face, default 5 in SCRFD
  */
  int landmarks_per_face;
  /*! @brief
  Argument for image postprocessing step, the outputs of onnx file with key
  points features or not, default true
  */
  bool use_kps;
  /*! @brief
  Argument for image postprocessing step, the upperbond number of boxes
  processed by nms, default 30000
  */
  int max_nms;
  /*! @brief
  Argument for image postprocessing step, anchor number of each stride, default
  2
  */
  unsigned int num_anchors;

  /// This function will disable normalize and hwc2chw in preprocessing step.
  void DisableNormalize();

  /// This function will disable hwc2chw in preprocessing step.
  void DisablePermute();

private:
  bool Initialize();

  bool Preprocess(Mat *mat, FDTensor *output,
                  std::map<std::string, std::array<float, 2>> *im_info);

  bool Postprocess(std::vector<FDTensor> &infer_result,
                   FaceDetectionResult *result,
                   const std::map<std::string, std::array<float, 2>> &im_info,
                   float conf_threshold, float nms_iou_threshold);

  void GeneratePoints();

  void LetterBox(Mat *mat, const std::vector<int> &size,
                 const std::vector<float> &color, bool _auto,
                 bool scale_fill = false, bool scale_up = true,
                 int stride = 32);

  bool is_dynamic_input_;

  bool center_points_is_update_;

  typedef struct {
    float cx;
    float cy;
  } SCRFDPoint;

  std::unordered_map<int, std::vector<SCRFDPoint>> center_points_;

  // for recording the switch of normalize
  bool disable_normalize_ = false;
  // for recording the switch of hwc2chw
  bool disable_permute_ = false;
};
} // namespace facedet
} // namespace vision
} // namespace ultra_infer
