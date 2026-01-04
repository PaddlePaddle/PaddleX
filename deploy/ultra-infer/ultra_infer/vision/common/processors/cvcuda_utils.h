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
#include "ultra_infer/vision/common/processors/mat.h"

#ifdef ENABLE_CVCUDA
#include <cvcuda/Types.h>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>

namespace ultra_infer {
namespace vision {

nvcv::ImageFormat CreateCvCudaImageFormat(FDDataType type, int channel,
                                          bool interleaved = true);
std::shared_ptr<nvcv::TensorWrapData>
CreateCvCudaTensorWrapData(const FDTensor &tensor, Layout layout = Layout::HWC);
void *GetCvCudaTensorDataPtr(const nvcv::TensorWrapData &tensor);
nvcv::ImageWrapData CreateImageWrapData(const FDTensor &tensor);
void CreateCvCudaImageBatchVarShape(std::vector<FDTensor *> &tensors,
                                    nvcv::ImageBatchVarShape &img_batch);
NVCVInterpolationType CreateCvCudaInterp(int interp);

} // namespace vision
} // namespace ultra_infer
#endif
