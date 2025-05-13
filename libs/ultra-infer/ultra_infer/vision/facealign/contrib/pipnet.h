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

namespace facealign {
/*! @brief PIPNet model object used when to load a PIPNet model exported by
 * PIPNet.
 */
class ULTRAINFER_DECL PIPNet : public UltraInferModel {
public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./pipnet.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in "valid_cpu_backends" \param[in] model_format
   * Model format of the loaded model, default is ONNX format
   */
  PIPNet(const std::string &model_file, const std::string &params_file = "",
         const RuntimeOption &custom_option = RuntimeOption(),
         const ModelFormat &model_format = ModelFormat::ONNX);

  std::string ModelName() const { return "PIPNet"; }
  /** \brief Predict the face detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array
   * with layout HWC, BGR format \param[in] result The output face detection
   * result will be written to this structure \return true if the prediction
   * successed, otherwise false
   */
  virtual bool Predict(cv::Mat *im, FaceAlignmentResult *result);

  /** \brief Get the number of landmakrs
   *
   * \return Integer type, default num_landmarks = 19
   */
  int GetNumLandmarks() { return num_landmarks_; }
  /** \brief Get the mean values for normalization
   *
   * \return Vector of float values, default mean_vals = {0.485f, 0.456f,
   * 0.406f}
   */
  std::vector<float> GetMeanVals() { return mean_vals_; }
  /** \brief Get the std values for normalization
   *
   * \return Vector of float values, default std_vals = {0.229f, 0.224f, 0.225f}
   */
  std::vector<float> GetStdVals() { return std_vals_; }
  /** \brief Get the input size of image
   *
   * \return Vector of int values, default {256, 256}
   */
  std::vector<int> GetSize() { return size_; }
  /** \brief Set the number of landmarks
   *
   * \param[in] num_landmarks Integer value which represents number of landmarks
   */
  void SetNumLandmarks(const int &num_landmarks);
  /** \brief Set the mean values for normalization
   *
   * \param[in] mean_vals Vector of float values whose length is equal to 3
   */
  void SetMeanVals(const std::vector<float> &mean_vals) {
    mean_vals_ = mean_vals;
  }
  /** \brief Set the std values for normalization
   *
   * \param[in] std_vals Vector of float values whose length is equal to 3
   */
  void SetStdVals(const std::vector<float> &std_vals) { std_vals_ = std_vals; }
  /** \brief Set the input size of image
   *
   * \param[in] size Vector of int values which represents {width, height} of
   * image
   */
  void SetSize(const std::vector<int> &size) { size_ = size; }

private:
  bool Initialize();

  bool Preprocess(Mat *mat, FDTensor *outputs,
                  std::map<std::string, std::array<int, 2>> *im_info);

  bool Postprocess(std::vector<FDTensor> &infer_result,
                   FaceAlignmentResult *result,
                   const std::map<std::string, std::array<int, 2>> &im_info);
  void GenerateLandmarks(std::vector<FDTensor> &infer_result,
                         FaceAlignmentResult *result, float img_height,
                         float img_width);
  std::map<int, int> num_lms_map_;
  std::map<int, int> max_len_map_;
  std::map<int, std::vector<int>> reverse_index1_map_;
  std::map<int, std::vector<int>> reverse_index2_map_;
  int num_nb_;
  int net_stride_;
  // Now PIPNet support num_landmarks in {19, 29, 68, 98}
  std::vector<int> supported_num_landmarks_;
  // tuple of (width, height), default (256, 256)
  std::vector<int> size_;

  // Mean parameters for normalize, size should be the the same as channels,
  // default mean_vals = {0.485f, 0.456f, 0.406f}
  std::vector<float> mean_vals_;
  // Std parameters for normalize, size should be the the same as channels,
  // default std_vals = {0.229f, 0.224f, 0.225f}
  std::vector<float> std_vals_;
  // number of landmarks
  int num_landmarks_;
};

} // namespace facealign
} // namespace vision
} // namespace ultra_infer
