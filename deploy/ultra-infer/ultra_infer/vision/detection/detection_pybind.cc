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

void BindYOLOv7(pybind11::module &m);
void BindScaledYOLOv4(pybind11::module &m);
void BindYOLOR(pybind11::module &m);
void BindYOLOv6(pybind11::module &m);
void BindYOLOv5Lite(pybind11::module &m);
void BindYOLOv5(pybind11::module &m);
void BindYOLOv5Seg(pybind11::module &m);
void BindFastestDet(pybind11::module &m);
void BindYOLOX(pybind11::module &m);
void BindNanoDetPlus(pybind11::module &m);
void BindPPDet(pybind11::module &m);
void BindYOLOv7End2EndTRT(pybind11::module &m);
void BindYOLOv7End2EndORT(pybind11::module &m);
void BindYOLOv8(pybind11::module &m);
void BindRKYOLO(pybind11::module &m);

void BindDetection(pybind11::module &m) {
  auto detection_module =
      m.def_submodule("detection", "Image object detection models.");
  BindPPDet(detection_module);
  BindYOLOv7(detection_module);
  BindScaledYOLOv4(detection_module);
  BindYOLOR(detection_module);
  BindYOLOv6(detection_module);
  BindYOLOv5Lite(detection_module);
  BindYOLOv5(detection_module);
  BindYOLOv5Seg(detection_module);
  BindFastestDet(detection_module);
  BindYOLOX(detection_module);
  BindNanoDetPlus(detection_module);
  BindYOLOv7End2EndTRT(detection_module);
  BindYOLOv7End2EndORT(detection_module);
  BindYOLOv8(detection_module);
  BindRKYOLO(detection_module);
}
} // namespace ultra_infer
