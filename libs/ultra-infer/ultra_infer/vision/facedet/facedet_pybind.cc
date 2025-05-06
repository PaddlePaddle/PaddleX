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

void BindRetinaFace(pybind11::module &m);
void BindUltraFace(pybind11::module &m);
void BindYOLOv5Face(pybind11::module &m);
void BindYOLOv7Face(pybind11::module &m);
void BindCenterFace(pybind11::module &m);
void BindBlazeFace(pybind11::module &m);
void BindSCRFD(pybind11::module &m);

void BindFaceDet(pybind11::module &m) {
  auto facedet_module = m.def_submodule("facedet", "Face detection models.");
  BindRetinaFace(facedet_module);
  BindUltraFace(facedet_module);
  BindYOLOv5Face(facedet_module);
  BindYOLOv7Face(facedet_module);
  BindCenterFace(facedet_module);
  BindBlazeFace(facedet_module);
  BindSCRFD(facedet_module);
}
} // namespace ultra_infer
