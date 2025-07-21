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

#include "ultra_infer/pybind/main.h"

namespace ultra_infer {

void BindPFLD(pybind11::module &m);
void BindFaceLandmark1000(pybind11::module &m);
void BindPIPNet(pybind11::module &m);

void BindFaceAlign(pybind11::module &m) {
  auto facedet_module = m.def_submodule("facealign", "Face alignment models.");
  BindPFLD(facedet_module);
  BindFaceLandmark1000(facedet_module);
  BindPIPNet(facedet_module);
}
} // namespace ultra_infer
