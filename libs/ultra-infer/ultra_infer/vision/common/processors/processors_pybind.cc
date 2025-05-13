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

void BindProcessorManager(pybind11::module &m);
void BindNormalizeAndPermute(pybind11::module &m);
void BindProcessor(pybind11::module &m);
void BindResizeByShort(pybind11::module &m);
void BindCenterCrop(pybind11::module &m);
void BindPad(pybind11::module &m);
void BindCast(pybind11::module &m);
void BindHWC2CHW(pybind11::module &m);
void BindNormalize(pybind11::module &m);
void BindPadToSize(pybind11::module &m);
void BindResize(pybind11::module &m);
void BindStridePad(pybind11::module &m);

void BindProcessors(pybind11::module &m) {
  auto processors_m =
      m.def_submodule("processors", "Module to deploy Processors models");
  BindProcessorManager(processors_m);
  BindProcessor(processors_m);
  BindNormalizeAndPermute(processors_m);
  BindResizeByShort(processors_m);
  BindCenterCrop(processors_m);
  BindPad(processors_m);
  BindCast(processors_m);
  BindHWC2CHW(processors_m);
  BindNormalize(processors_m);
  BindPadToSize(processors_m);
  BindResize(processors_m);
  BindStridePad(processors_m);
}
} // namespace ultra_infer
