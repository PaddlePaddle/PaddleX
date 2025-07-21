// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.  //NOLINT
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
#include "ultra_infer/vision/faceid/contrib/insightface/base.h"

namespace ultra_infer {
namespace vision {
namespace faceid {
class ULTRAINFER_DECL ArcFace : public InsightFaceRecognitionBase {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g ArcFace/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g ArcFace/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in `valid_cpu_backends` \param[in] model_format
   * Model format of the loaded model, default is Paddle format
   */
  ArcFace(const std::string &model_file, const std::string &params_file = "",
          const RuntimeOption &custom_option = RuntimeOption(),
          const ModelFormat &model_format = ModelFormat::ONNX)
      : InsightFaceRecognitionBase(model_file, params_file, custom_option,
                                   model_format) {
    if (model_format == ModelFormat::ONNX) {
      valid_cpu_backends = {Backend::ORT};
      valid_gpu_backends = {Backend::ORT, Backend::TRT};
    } else if (model_format == ModelFormat::RKNN) {
      valid_rknpu_backends = {Backend::RKNPU2};
    } else {
      valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::LITE};
      valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
      valid_kunlunxin_backends = {Backend::LITE};
    }
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "ArcFace"; }
};

class ULTRAINFER_DECL CosFace : public InsightFaceRecognitionBase {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g CosFace/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g CosFace/model.pdiparams,
   * if the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in `valid_cpu_backends` \param[in] model_format
   * Model format of the loaded model, default is Paddle format
   */
  CosFace(const std::string &model_file, const std::string &params_file = "",
          const RuntimeOption &custom_option = RuntimeOption(),
          const ModelFormat &model_format = ModelFormat::ONNX)
      : InsightFaceRecognitionBase(model_file, params_file, custom_option,
                                   model_format) {
    if (model_format == ModelFormat::ONNX) {
      valid_cpu_backends = {Backend::ORT};
      valid_gpu_backends = {Backend::ORT, Backend::TRT};
    } else if (model_format == ModelFormat::RKNN) {
      valid_rknpu_backends = {Backend::RKNPU2};
    } else {
      valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::LITE};
      valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
      valid_kunlunxin_backends = {Backend::LITE};
    }
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "CosFace"; }
};
class ULTRAINFER_DECL PartialFC : public InsightFaceRecognitionBase {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g PartialFC/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g
   * PartialFC/model.pdiparams, if the model format is ONNX, this parameter will
   * be ignored \param[in] custom_option RuntimeOption for inference, the
   * default will use cpu, and choose the backend defined in
   * `valid_cpu_backends` \param[in] model_format Model format of the loaded
   * model, default is Paddle format
   */
  PartialFC(const std::string &model_file, const std::string &params_file = "",
            const RuntimeOption &custom_option = RuntimeOption(),
            const ModelFormat &model_format = ModelFormat::ONNX)
      : InsightFaceRecognitionBase(model_file, params_file, custom_option,
                                   model_format) {
    if (model_format == ModelFormat::ONNX) {
      valid_cpu_backends = {Backend::ORT};
      valid_gpu_backends = {Backend::ORT, Backend::TRT};
    } else if (model_format == ModelFormat::RKNN) {
      valid_rknpu_backends = {Backend::RKNPU2};
    } else {
      valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::LITE};
      valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
      valid_kunlunxin_backends = {Backend::LITE};
    }
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PartialFC"; }
};
class ULTRAINFER_DECL VPL : public InsightFaceRecognitionBase {
public:
  /** \brief Set path of model file and configuration file, and the
   * configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g VPL/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g VPL/model.pdiparams, if
   * the model format is ONNX, this parameter will be ignored \param[in]
   * custom_option RuntimeOption for inference, the default will use cpu, and
   * choose the backend defined in `valid_cpu_backends` \param[in] model_format
   * Model format of the loaded model, default is Paddle format
   */
  VPL(const std::string &model_file, const std::string &params_file = "",
      const RuntimeOption &custom_option = RuntimeOption(),
      const ModelFormat &model_format = ModelFormat::ONNX)
      : InsightFaceRecognitionBase(model_file, params_file, custom_option,
                                   model_format) {
    if (model_format == ModelFormat::ONNX) {
      valid_cpu_backends = {Backend::ORT};
      valid_gpu_backends = {Backend::ORT, Backend::TRT};
    } else if (model_format == ModelFormat::RKNN) {
      valid_rknpu_backends = {Backend::RKNPU2};
    } else {
      valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::LITE};
      valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
      valid_kunlunxin_backends = {Backend::LITE};
    }
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "VPL"; }
};

} // namespace faceid
} // namespace vision
} // namespace ultra_infer
