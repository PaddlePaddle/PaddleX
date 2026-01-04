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
#include "ultra_infer/vision/common/processors/manager.h"
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"

namespace ultra_infer {
namespace vision {

namespace perception {
/*! @brief Preprocessor object for Centerpoint model.
 */
class ULTRAINFER_DECL CenterpointPreprocessor : public ProcessorManager {
public:
  CenterpointPreprocessor() = default;
  /** \brief Create a preprocessor instance for Centerpoint model
   *
   * \param[in] config_file Path of configuration file for deployment, e.g
   * Centerpoint/infer_cfg.yml
   */
  explicit CenterpointPreprocessor(const std::string &config_file);

  bool Apply(FDMatBatch *image_batch, std::vector<FDTensor> *outputs) {
    return false;
  }

  bool Apply(std::vector<std::string> &points_dir, const int64_t num_point_dim,
             const int with_timelag, std::vector<FDTensor> &outputs);

  bool Run(std::vector<std::string> &points_dir, const int64_t num_point_dim,
           const int with_timelag, std::vector<FDTensor> &outputs);

protected:
  std::vector<std::shared_ptr<Processor>> processors_;
  bool ReadPoint(const std::string &file_path, const int64_t num_point_dim,
                 std::vector<float> &data, int64_t *num_points);
  bool InsertTimeToPoints(const int64_t num_points, const int64_t num_point_dim,
                          float *points);
  bool initialized_ = false;
};

} // namespace perception
} // namespace vision
} // namespace ultra_infer
