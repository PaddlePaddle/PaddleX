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
#include "ultra_infer/vision/generation/contrib/postprocessor.h"
#include "ultra_infer/vision/generation/contrib/preprocessor.h"

namespace ultra_infer {

namespace vision {

namespace generation {
/*! @brief AnimeGAN model object is used when load a AnimeGAN model.
 */
class ULTRAINFER_DECL AnimeGAN : public UltraInferModel {
public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./model.pdmodel
   * \param[in] params_file Path of parameter file, e.g ./model.pdiparams, if
   * the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in "valid_cpu_backends" \param[in] model_format
   * Model format of the loaded model, default is PADDLE format
   */
  AnimeGAN(const std::string &model_file, const std::string &params_file = "",
           const RuntimeOption &custom_option = RuntimeOption(),
           const ModelFormat &model_format = ModelFormat::PADDLE);

  std::string ModelName() const { return "styletransfer/animegan"; }

  /** \brief Predict the style transfer result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output style transfer
   * result will be written to this structure \return true if the prediction
   * succeeded, otherwise false
   */
  bool Predict(cv::Mat &img, cv::Mat *result);

  /** \brief Predict the style transfer result for a batch of input images
   *
   * \param[in] images The list of input images, each element comes from
   * cv::imread(), is a 3-D array with layout HWC, BGR format \param[in] results
   * The list of output style transfer results will be written to this structure
   * \return true if the batch prediction succeeded, otherwise false
   */
  bool BatchPredict(const std::vector<cv::Mat> &images,
                    std::vector<cv::Mat> *results);

  // Get preprocessor reference of AnimeGAN
  AnimeGANPreprocessor &GetPreprocessor() { return preprocessor_; }

  // Get postprocessor reference of AnimeGAN
  AnimeGANPostprocessor &GetPostprocessor() { return postprocessor_; }

private:
  bool Initialize();

  AnimeGANPreprocessor preprocessor_;
  AnimeGANPostprocessor postprocessor_;
};

} // namespace generation
} // namespace vision
} // namespace ultra_infer
