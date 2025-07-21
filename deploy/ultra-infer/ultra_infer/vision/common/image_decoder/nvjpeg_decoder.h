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
//
// Part of the following code in this file refs to
// https://github.com/CVCUDA/CV-CUDA/blob/release_v0.2.x/samples/common/NvDecoder.h
//
// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the Apache-2.0 license
// \brief
// \author NVIDIA

#pragma once

#ifdef ENABLE_NVJPEG
#include "ultra_infer/core/fd_tensor.h"

#include <cuda_runtime_api.h>
#include <nvjpeg.h>

namespace ultra_infer {
namespace vision {
namespace nvjpeg {

typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector<char>> FileData;

struct decode_params_t {
  int batch_size;
  nvjpegJpegState_t nvjpeg_state;
  nvjpegHandle_t nvjpeg_handle;
  cudaStream_t stream;

  // used with decoupled API
  nvjpegJpegState_t nvjpeg_decoupled_state;
  nvjpegBufferPinned_t pinned_buffers[2]; // 2 buffers for pipelining
  nvjpegBufferDevice_t device_buffer;
  nvjpegJpegStream_t jpeg_streams[2]; // 2 streams for pipelining
  nvjpegDecodeParams_t nvjpeg_decode_params;
  nvjpegJpegDecoder_t nvjpeg_decoder;

  nvjpegOutputFormat_t fmt;
  bool hw_decode_available;
};

void init_decoder(decode_params_t &params);
void destroy_decoder(decode_params_t &params);

double process_images(const FileNames &image_names, decode_params_t &params,
                      double &total, std::vector<nvjpegImage_t> &iout,
                      std::vector<FDTensor *> &output_buffers,
                      std::vector<int> &widths, std::vector<int> &heights);

} // namespace nvjpeg
} // namespace vision
} // namespace ultra_infer

#endif // ENABLE_NVJPEG
