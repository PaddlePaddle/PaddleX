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

#include <vector>

#include "ultra_infer/ultra_infer_model.h"
#include "ultra_infer/vision/common/processors/transform.h"
#include "ultra_infer/vision/common/result.h"

#include "ultra_infer/utils/unique_ptr.h"
#include "ultra_infer/vision/ocr/ppocr/structurev2_layout.h"

namespace ultra_infer {

namespace pipeline {
typedef ultra_infer::vision::ocr::StructureV2Layout PPStructureV2Layout;

namespace application {
namespace ocrsystem {

// TODO(qiuyanjun): This pipeline may not need
typedef pipeline::PPStructureV2Layout PPStructureV2LayoutSystem;
} // namespace ocrsystem
} // namespace application

} // namespace pipeline
} // namespace ultra_infer
