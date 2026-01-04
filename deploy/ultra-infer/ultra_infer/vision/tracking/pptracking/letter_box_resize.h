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

#include "ultra_infer/vision/common/processors/base.h"

namespace ultra_infer {
namespace vision {

class LetterBoxResize : public Processor {
public:
  LetterBoxResize(const std::vector<int> &target_size,
                  const std::vector<float> &color) {
    target_size_ = target_size;
    color_ = color;
  }

  std::string Name() override { return "LetterBoxResize"; }
  bool ImplByOpenCV(Mat *mat) override;
#ifdef ENABLE_FLYCV
  bool ImplByFlyCV(FDMat *mat) override;
#endif
#ifdef ENABLE_CVCUDA
  virtual bool ImplByCvCuda(FDMat *mat) override;
#endif

#ifdef ENABLE_CUDA
  virtual bool ImplByCuda(FDMat *mat);
#endif

  static bool Run(Mat *mat, const std::vector<int> &target_size,
                  const std::vector<float> &color,
                  ProcLib lib = ProcLib::DEFAULT);

private:
  std::vector<int> target_size_;
  std::vector<float> color_;
};
} // namespace vision
} // namespace ultra_infer
