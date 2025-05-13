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

#include "ultra_infer/vision/keypointdet/pptinypose/pptinypose_utils.h"

namespace ultra_infer {
namespace vision {
/** \brief All keypoint detection model APIs are defined inside this namespace
 *
 */
namespace keypointdetection {

/*! @brief PPTinyPose model object used when to load a PPTinyPose model exported
 * by PaddleDetection
 */
class ULTRAINFER_DECL PPTinyPose : public UltraInferModel {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g pptinypose/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g
   * pptinypose/model.pdiparams, if the model format is ONNX, this parameter
   * will be ignored \param[in] config_file Path of configuration file for
   * deployment, e.g pptinypose/infer_cfg.yml \param[in] custom_option
   * RuntimeOption for inference, the default will use cpu, and choose the
   * backend defined in `valid_cpu_backends` \param[in] model_format Model
   * format of the loaded model, default is Paddle format
   */
  PPTinyPose(const std::string &model_file, const std::string &params_file,
             const std::string &config_file,
             const RuntimeOption &custom_option = RuntimeOption(),
             const ModelFormat &model_format = ModelFormat::PADDLE);

  /// Get model's name
  std::string ModelName() const { return "PaddleDetection/PPTinyPose"; }

  /** \brief Predict the keypoint detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread()
   * \param[in] result The output keypoint detection result will be written to
   * this structure \return true if the keypoint prediction successed, otherwise
   * false
   */
  bool Predict(cv::Mat *im, KeyPointDetectionResult *result);

  /** \brief Predict the keypoint detection result with given detection result
   * for an input image
   *
   * \param[in] im The input image data, comes from cv::imread()
   * \param[in] result The output keypoint detection result will be written to
   * this structure \param[in] detection_result The structure stores pedestrian
   * detection result, which is used to crop image for multi-persons keypoint
   * detection \return true if the keypoint prediction successed, otherwise
   * false
   */
  bool Predict(cv::Mat *im, KeyPointDetectionResult *result,
               const DetectionResult &detection_result);

  /** \brief Whether using Distribution-Aware Coordinate Representation for
   * Human Pose Estimation(DARK for short) in postprocess, default is true
   */
  bool use_dark = true;

  /// This function will disable normalize in preprocessing step.
  void DisableNormalize() {
    disable_normalize_ = true;
    BuildPreprocessPipelineFromConfig();
  }

  /// This function will disable hwc2chw in preprocessing step.
  void DisablePermute() {
    disable_permute_ = true;
    BuildPreprocessPipelineFromConfig();
  }

protected:
  bool Initialize();
  /// Build the preprocess pipeline from the loaded model
  bool BuildPreprocessPipelineFromConfig();
  /// Preprocess an input image, and set the preprocessed results to `outputs`
  bool Preprocess(Mat *mat, std::vector<FDTensor> *outputs);

  /// Postprocess the inferenced results, and set the final result to `result`
  bool Postprocess(std::vector<FDTensor> &infer_result,
                   KeyPointDetectionResult *result,
                   const std::vector<float> &center,
                   const std::vector<float> &scale);

private:
  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
  // for recording the switch of hwc2chw
  bool disable_permute_ = false;
  // for recording the switch of normalize
  bool disable_normalize_ = false;
};
} // namespace keypointdetection
} // namespace vision
} // namespace ultra_infer
