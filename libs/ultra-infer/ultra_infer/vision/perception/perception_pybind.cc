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

void BindSmoke(pybind11::module &m);
void BindPetr(pybind11::module &m);
void BindCenterpoint(pybind11::module &m);
void BindCaddn(pybind11::module &m);

void BindPerception(pybind11::module &m) {
  auto perception_module =
      m.def_submodule("perception", "3D object perception models.");
  BindSmoke(perception_module);
  BindPetr(perception_module);
  BindCenterpoint(perception_module);
  BindCaddn(perception_module);
}
} // namespace ultra_infer
