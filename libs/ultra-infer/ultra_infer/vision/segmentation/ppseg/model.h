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
#include "ultra_infer/vision/segmentation/ppseg/postprocessor.h"
#include "ultra_infer/vision/segmentation/ppseg/preprocessor.h"

namespace ultra_infer {
namespace vision {
/** \brief All segmentation model APIs are defined inside this namespace
 *
 */
namespace segmentation {

/*! @brief PaddleSeg serials model object used when to load a PaddleSeg model
 * exported by PaddleSeg repository
 */
class ULTRAINFER_DECL PaddleSegModel : public UltraInferModel {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g unet/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g unet/model.pdiparams, if
   * the model format is ONNX, this parameter will be ignored \param[in]
   * config_file Path of configuration file for deployment, e.g unet/deploy.yml
   * \param[in] custom_option RuntimeOption for inference, the default will use
   * cpu, and choose the backend defined in `valid_cpu_backends` \param[in]
   * model_format Model format of the loaded model, default is Paddle format
   */
  PaddleSegModel(const std::string &model_file, const std::string &params_file,
                 const std::string &config_file,
                 const RuntimeOption &custom_option = RuntimeOption(),
                 const ModelFormat &model_format = ModelFormat::PADDLE);

  /** \brief Clone a new PaddleSegModel with less memory usage when multiple
   * instances of the same model are created
   *
   * \return new PaddleDetModel* type unique pointer
   */
  virtual std::unique_ptr<PaddleSegModel> Clone() const;

  /// Get model's name
  std::string ModelName() const { return "PaddleSeg"; }

  /** \brief DEPRECATED Predict the segmentation result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output segmentation
   * result will be written to this structure \return true if the segmentation
   * prediction successed, otherwise false
   */
  virtual bool Predict(cv::Mat *im, SegmentationResult *result);

  /** \brief Predict the segmentation result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output segmentation
   * result will be written to this structure \return true if the segmentation
   * prediction successed, otherwise false
   */
  virtual bool Predict(const cv::Mat &im, SegmentationResult *result);

  /** \brief Predict the segmentation results for a batch of input images
   *
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output segmentation result list
   * \return true if the prediction successed, otherwise false
   */
  virtual bool BatchPredict(const std::vector<cv::Mat> &imgs,
                            std::vector<SegmentationResult> *results);

  /// Get preprocessor reference of PaddleSegModel
  virtual PaddleSegPreprocessor &GetPreprocessor() { return preprocessor_; }

  /// Get postprocessor reference of PaddleSegModel
  virtual PaddleSegPostprocessor &GetPostprocessor() { return postprocessor_; }

protected:
  bool Initialize();
  PaddleSegPreprocessor preprocessor_;
  PaddleSegPostprocessor postprocessor_;
};

} // namespace segmentation
} // namespace vision
} // namespace ultra_infer
