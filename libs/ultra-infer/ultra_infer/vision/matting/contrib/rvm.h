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
/** \brief All image/video matting model APIs are defined inside this namespace
 *
 */
namespace matting {

/*! @brief RobustVideoMatting model object used when to load a
 * RobustVideoMatting model exported by RobustVideoMatting
 */
class ULTRAINFER_DECL RobustVideoMatting : public UltraInferModel {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g rvm/rvm_mobilenetv3_fp32.onnx
   * \param[in] params_file Path of parameter file, if the model format is ONNX,
   * this parameter will be ignored \param[in] custom_option RuntimeOption for
   * inference, the default will use cpu, and choose the backend defined in
   * `valid_cpu_backends` \param[in] model_format Model format of the loaded
   * model, default is ONNX format
   */
  RobustVideoMatting(const std::string &model_file,
                     const std::string &params_file = "",
                     const RuntimeOption &custom_option = RuntimeOption(),
                     const ModelFormat &model_format = ModelFormat::ONNX);

  /// Get model's name
  std::string ModelName() const { return "matting/RobustVideoMatting"; }

  /** \brief Predict the matting result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread()
   * \param[in] result The output matting result will be written to this
   * structure \return true if the prediction successed, otherwise false
   */
  bool Predict(cv::Mat *im, MattingResult *result);

  /// Preprocess image size, the default is (1080, 1920)
  std::vector<int> size;

  /// Whether to open the video mode, if there are some irrelevant pictures, set
  /// it to false, the default is true // NOLINT
  bool video_mode;

  /// Whether convert to RGB, Set to false if you have converted YUV format
  /// images to RGB outside the model, default true // NOLINT
  bool swap_rb;

private:
  bool Initialize();
  /// Preprocess an input image, and set the preprocessed results to `outputs`
  bool Preprocess(Mat *mat, FDTensor *output,
                  std::map<std::string, std::array<int, 2>> *im_info);

  /// Postprocess the inferenced results, and set the final result to `result`
  bool Postprocess(std::vector<FDTensor> &infer_result, MattingResult *result,
                   const std::map<std::string, std::array<int, 2>> &im_info);

  /// Init dynamic inputs datas
  std::vector<std::vector<float>> dynamic_inputs_datas_ = {
      {0.0f},  // r1i
      {0.0f},  // r2i
      {0.0f},  // r3i
      {0.0f},  // r4i
      {0.25f}, // downsample_ratio
  };

  /// Init dynamic inputs dims
  std::vector<std::vector<int64_t>> dynamic_inputs_dims_ = {
      {1, 1, 1, 1}, // r1i
      {1, 1, 1, 1}, // r2i
      {1, 1, 1, 1}, // r3i
      {1, 1, 1, 1}, // r4i
      {1},          // downsample_ratio
  };
};

} // namespace matting
} // namespace vision
} // namespace ultra_infer
