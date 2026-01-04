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
#include "ultra_infer/vision/sr/ppsr/ppmsvsr.h"

namespace ultra_infer {
namespace vision {
namespace sr {

class ULTRAINFER_DECL EDVR : public PPMSVSR {
public:
  /**
   * Set path of model file and configuration file, and the configuration of
   * runtime
   * @param[in] model_file Path of model file, e.g EDVR/model.pdmodel
   * @param[in] params_file Path of parameter file, e.g EDVR/model.pdiparams
   * @param[in] custom_option RuntimeOption for inference, the default will use
   * cpu, and choose the backend defined in `valid_cpu_backends`
   * @param[in] model_format Model format of the loaded model, default is Paddle
   * format
   */
  EDVR(const std::string &model_file, const std::string &params_file,
       const RuntimeOption &custom_option = RuntimeOption(),
       const ModelFormat &model_format = ModelFormat::PADDLE);
  /// model name contained EDVR
  std::string ModelName() const override { return "EDVR"; }

private:
  bool Postprocess(std::vector<FDTensor> &infer_results,
                   std::vector<cv::Mat> &results) override;
};
} // namespace sr
} // namespace vision
} // namespace ultra_infer
