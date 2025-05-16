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
#include "ultra_infer/vision/classification/ppshitu/ppshituv2_rec_postprocessor.h"
#include "ultra_infer/vision/classification/ppshitu/ppshituv2_rec_preprocessor.h"

namespace ultra_infer {
namespace vision {
namespace classification {
/*! @brief PPShiTuV2Recognizer model object used when to load a
 * PPShiTuV2Recognizer model exported by PP-ShiTuV2 Rec model.
 */
class ULTRAINFER_DECL PPShiTuV2Recognizer : public UltraInferModel {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g PPLCNet/inference.pdmodel
   * \param[in] params_file Path of parameter file, e.g
   * PPLCNet/inference.pdiparams, if the model format is ONNX, this parameter
   * will be ignored \param[in] config_file Path of configuration file for
   * deployment, e.g PPLCNet/inference_cls.yml \param[in] custom_option
   * RuntimeOption for inference, the default will use cpu, and choose the
   * backend defined in `valid_cpu_backends` \param[in] model_format Model
   * format of the loaded model, default is Paddle format
   */
  PPShiTuV2Recognizer(const std::string &model_file,
                      const std::string &params_file,
                      const std::string &config_file,
                      const RuntimeOption &custom_option = RuntimeOption(),
                      const ModelFormat &model_format = ModelFormat::PADDLE);

  /** \brief Clone a new PPShiTuV2Recognizer with less memory usage when
   * multiple instances of the same model are created
   *
   * \return new PPShiTuV2Recognizer* type unique pointer
   */
  virtual std::unique_ptr<PPShiTuV2Recognizer> Clone() const;

  /// Get model's name
  virtual std::string ModelName() const { return "PPShiTuV2Recognizer"; }

  /** \brief DEPRECATED Predict the feature vector result for an input image,
   * remove at 1.0 version
   *
   * \param[in] im The input image data, comes from cv::imread()
   * \param[in] result The output feature vector result will be written to this
   * structure \return true if the prediction succeeded, otherwise false
   */
  virtual bool Predict(cv::Mat *im, ClassifyResult *result);

  /** \brief Predict the classification result for an input image
   *
   * \param[in] img The input image data, comes from cv::imread()
   * \param[in] result The output feature vector result
   * \return true if the prediction succeeded, otherwise false
   */
  virtual bool Predict(const cv::Mat &img, ClassifyResult *result);

  /** \brief Predict the feature vector results for a batch of input images
   *
   * \param[in] imgs, The input image list, each element comes from cv::imread()
   * \param[in] results The output feature vector(namely ClassifyResult.feature)
   * result list \return true if the prediction succeeded, otherwise false
   */
  virtual bool BatchPredict(const std::vector<cv::Mat> &imgs,
                            std::vector<ClassifyResult> *results);

  /** \brief Predict the feature vector result for an input image
   *
   * \param[in] mat The input mat
   * \param[in] result The output feature vector result
   * \return true if the prediction succeeded, otherwise false
   */
  virtual bool Predict(const FDMat &mat, ClassifyResult *result);

  /** \brief Predict the feature vector results for a batch of input images
   *
   * \param[in] mats, The input mat list
   * \param[in] results The output feature vector result list
   * \return true if the prediction succeeded, otherwise false
   */
  virtual bool BatchPredict(const std::vector<FDMat> &mats,
                            std::vector<ClassifyResult> *results);

  /// Get preprocessor reference of PPShiTuV2Recognizer
  virtual PPShiTuV2RecognizerPreprocessor &GetPreprocessor() {
    return preprocessor_;
  }

  /// Get postprocessor reference of PPShiTuV2Recognizer
  virtual PPShiTuV2RecognizerPostprocessor &GetPostprocessor() {
    return postprocessor_;
  }

protected:
  bool Initialize();
  PPShiTuV2RecognizerPreprocessor preprocessor_;
  PPShiTuV2RecognizerPostprocessor postprocessor_;
};

} // namespace classification
} // namespace vision
} // namespace ultra_infer
