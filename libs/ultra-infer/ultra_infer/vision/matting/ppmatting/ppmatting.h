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
/** \brief All object matting model APIs are defined inside this namespace
 *
 */
namespace matting {
/*! @brief PPMatting model object used when to load a PPMatting model exported
 * by PPMatting.
 */
class ULTRAINFER_DECL PPMatting : public UltraInferModel {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g PPMatting-512/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g
   * PPMatting-512/model.pdiparams, if the model format is ONNX, this parameter
   * will be ignored \param[in] config_file Path of configuration file for
   * deployment, e.g PPMatting-512/infer_cfg.yml \param[in] custom_option
   * RuntimeOption for inference, the default will use cpu, and choose the
   * backend defined in `valid_cpu_backends` \param[in] model_format Model
   * format of the loaded model, default is Paddle format
   */
  PPMatting(const std::string &model_file, const std::string &params_file,
            const std::string &config_file,
            const RuntimeOption &custom_option = RuntimeOption(),
            const ModelFormat &model_format = ModelFormat::PADDLE);

  std::string ModelName() const { return "PaddleMatting"; }
  /** \brief Predict the matting result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output matting result
   * will be written to this structure \return true if the prediction successed,
   * otherwise false
   */
  virtual bool Predict(cv::Mat *im, MattingResult *result);

private:
  bool Initialize();

  bool BuildPreprocessPipelineFromConfig();

  bool Preprocess(Mat *mat, FDTensor *outputs,
                  std::map<std::string, std::array<int, 2>> *im_info);

  bool Postprocess(std::vector<FDTensor> &infer_result, MattingResult *result,
                   const std::map<std::string, std::array<int, 2>> &im_info);

  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
  bool is_fixed_input_shape_;
};

} // namespace matting
} // namespace vision
} // namespace ultra_infer
