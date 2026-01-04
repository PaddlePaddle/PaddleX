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

void BindYOLOv5Cls(pybind11::module &m);
void BindPaddleClas(pybind11::module &m);
void BindPPShiTuV2(pybind11::module &m);
void BindResNet(pybind11::module &m);

void BindClassification(pybind11::module &m) {
  auto classification_module =
      m.def_submodule("classification", "Image classification models.");

  BindYOLOv5Cls(classification_module);
  BindPaddleClas(classification_module);
  BindPPShiTuV2(classification_module);
  BindResNet(classification_module);
}

} // namespace ultra_infer
