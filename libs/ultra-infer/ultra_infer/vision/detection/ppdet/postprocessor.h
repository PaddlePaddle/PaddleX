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
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"
#include "ultra_infer/vision/detection/ppdet/multiclass_nms.h"
#include "ultra_infer/vision/detection/ppdet/multiclass_nms_rotated.h"

namespace ultra_infer {
namespace vision {
namespace detection {
/*! @brief Postprocessor object for PaddleDet serials model.
 */
class ULTRAINFER_DECL PaddleDetPostprocessor {
public:
  PaddleDetPostprocessor() {
    // There may be no NMS config in the yaml file,
    // so we need to give a initial value to multi_class_nms_.
    multi_class_nms_.SetNMSOption(NMSOption());
    multi_class_nms_rotated_.SetNMSRotatedOption(NMSRotatedOption());
  }

  /** \brief Create a preprocessor instance for PaddleDet serials model
   *
   * \param[in] config_file Path of configuration file for deployment, e.g
   * ppyoloe/infer_cfg.yml
   */
  explicit PaddleDetPostprocessor(const std::string &arch) {
    // Used to differentiate models
    arch_ = arch;
    // There may be no NMS config in the yaml file,
    // so we need to give a initial value to multi_class_nms_.
    multi_class_nms_.SetNMSOption(NMSOption());
    multi_class_nms_rotated_.SetNMSRotatedOption(NMSRotatedOption());
  }

  /** \brief Process the result of runtime and fill to ClassifyResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of detection
   * \return true if the postprocess successed, otherwise false
   */
  bool Run(const std::vector<FDTensor> &tensors,
           std::vector<DetectionResult> *result);

  /// Apply box decoding and nms step for the outputs for the model.This is
  /// only available for those model exported without box decoding and nms.
  void ApplyNMS() { with_nms_ = false; }

  /// If you do not want to modify the Yaml configuration file,
  /// you can use this function to set rotated NMS parameters.
  void SetNMSRotatedOption(const NMSRotatedOption &option) {
    multi_class_nms_rotated_.SetNMSRotatedOption(option);
  }

  /// If you do not want to modify the Yaml configuration file,
  /// you can use this function to set NMS parameters.
  void SetNMSOption(const NMSOption &option) {
    multi_class_nms_.SetNMSOption(option);
  }

  // Set scale_factor_ value.This is only available for those model exported
  // without nms.
  void SetScaleFactor(const std::vector<float> &scale_factor_value) {
    scale_factor_ = scale_factor_value;
  }

private:
  std::vector<float> scale_factor_{0.0, 0.0};
  std::vector<float> GetScaleFactor() { return scale_factor_; }

  // for model without nms.
  bool with_nms_ = true;

  // Used to differentiate models
  std::string arch_;

  PaddleMultiClassNMS multi_class_nms_{};

  PaddleMultiClassNMSRotated multi_class_nms_rotated_{};

  // Process for General tensor without nms.
  bool ProcessWithoutNMS(const std::vector<FDTensor> &tensors,
                         std::vector<DetectionResult> *results);

  // Process for General tensor with nms.
  bool ProcessWithNMS(const std::vector<FDTensor> &tensors,
                      std::vector<DetectionResult> *results);

  // Process SOLOv2
  bool ProcessSolov2(const std::vector<FDTensor> &tensors,
                     std::vector<DetectionResult> *results);

  // Process PPYOLOER
  bool ProcessPPYOLOER(const std::vector<FDTensor> &tensors,
                       std::vector<DetectionResult> *results);

  // Process mask tensor for MaskRCNN
  bool ProcessMask(const FDTensor &tensor,
                   std::vector<DetectionResult> *results);
};

} // namespace detection
} // namespace vision
} // namespace ultra_infer
