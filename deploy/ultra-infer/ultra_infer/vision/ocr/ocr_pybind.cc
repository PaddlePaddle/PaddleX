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

void BindPPOCRModel(pybind11::module &m);
void BindPPOCRv4(pybind11::module &m);
void BindPPOCRv3(pybind11::module &m);
void BindPPOCRv2(pybind11::module &m);
void BindPPStructureV2Table(pybind11::module &m);

void BindOcr(pybind11::module &m) {
  auto ocr_module = m.def_submodule("ocr", "Module to deploy OCR models");
  BindPPOCRModel(ocr_module);
  BindPPOCRv4(ocr_module);
  BindPPOCRv3(ocr_module);
  BindPPOCRv2(ocr_module);
  BindPPStructureV2Table(ocr_module);
}
} // namespace ultra_infer
