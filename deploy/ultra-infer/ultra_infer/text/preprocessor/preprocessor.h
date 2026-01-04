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

#include "ultra_infer/core/fd_tensor.h"
#include "ultra_infer/utils/utils.h"
#include <memory>
#include <vector>

namespace ultra_infer {
namespace text {

class Preprocessor {
public:
  virtual bool Encode(const std::string &raw_text,
                      std::vector<FDTensor> *encoded_tensor) const;
  virtual bool EncodeBatch(const std::vector<std::string> &raw_texts,
                           std::vector<FDTensor> *encoded_tensor) const;
};

} // namespace text
} // namespace ultra_infer
