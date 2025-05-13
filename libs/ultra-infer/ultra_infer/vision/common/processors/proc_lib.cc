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

#include "ultra_infer/vision/common/processors/proc_lib.h"

namespace ultra_infer {
namespace vision {

ProcLib DefaultProcLib::default_lib = ProcLib::DEFAULT;

std::ostream &operator<<(std::ostream &out, const ProcLib &p) {
  switch (p) {
  case ProcLib::DEFAULT:
    out << "ProcLib::DEFAULT";
    break;
  case ProcLib::OPENCV:
    out << "ProcLib::OPENCV";
    break;
  case ProcLib::FLYCV:
    out << "ProcLib::FLYCV";
    break;
  case ProcLib::CUDA:
    out << "ProcLib::CUDA";
    break;
  case ProcLib::CVCUDA:
    out << "ProcLib::CVCUDA";
    break;
  default:
    FDASSERT(false, "Unknown type of ProcLib.");
  }
  return out;
}

} // namespace vision
} // namespace ultra_infer
