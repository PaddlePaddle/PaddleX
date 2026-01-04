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

#include "ultra_infer/utils/utils.h"
#include "ultra_infer/vision/common/image_decoder/nvjpeg_decoder.h"
#include "ultra_infer/vision/common/processors/mat.h"

namespace ultra_infer {
namespace vision {

enum class ULTRAINFER_DECL ImageDecoderLib { OPENCV, NVJPEG };

class ULTRAINFER_DECL ImageDecoder {
public:
  explicit ImageDecoder(ImageDecoderLib lib = ImageDecoderLib::OPENCV);

  ~ImageDecoder();

  bool Decode(const std::string &img_name, FDMat *mat);

  bool BatchDecode(const std::vector<std::string> &img_names,
                   std::vector<FDMat> *mats);

private:
  bool ImplByOpenCV(const std::vector<std::string> &img_names,
                    std::vector<FDMat> *mats);
  bool ImplByNvJpeg(const std::vector<std::string> &img_names,
                    std::vector<FDMat> *mats);
  ImageDecoderLib lib_ = ImageDecoderLib::OPENCV;
#ifdef ENABLE_NVJPEG
  nvjpeg::decode_params_t nvjpeg_params_;
#endif
};

} // namespace vision
} // namespace ultra_infer
