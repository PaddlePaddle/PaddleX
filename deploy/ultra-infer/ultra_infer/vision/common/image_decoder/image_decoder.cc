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

#include "ultra_infer/vision/common/image_decoder/image_decoder.h"

#include "opencv2/imgcodecs.hpp"

namespace ultra_infer {
namespace vision {

ImageDecoder::ImageDecoder(ImageDecoderLib lib) {
  if (lib == ImageDecoderLib::NVJPEG) {
#ifdef ENABLE_NVJPEG
    nvjpeg::init_decoder(nvjpeg_params_);
#endif
  }
  lib_ = lib;
}

ImageDecoder::~ImageDecoder() {
  if (lib_ == ImageDecoderLib::NVJPEG) {
#ifdef ENABLE_NVJPEG
    nvjpeg::destroy_decoder(nvjpeg_params_);
#endif
  }
}

bool ImageDecoder::Decode(const std::string &img_name, FDMat *mat) {
  std::vector<FDMat> mats(1);
  mats[0] = std::move(*mat);
  if (!BatchDecode({img_name}, &mats)) {
    return false;
  }
  *mat = std::move(mats[0]);
  return true;
}

bool ImageDecoder::BatchDecode(const std::vector<std::string> &img_names,
                               std::vector<FDMat> *mats) {
  if (lib_ == ImageDecoderLib::OPENCV) {
    return ImplByOpenCV(img_names, mats);
  } else if (lib_ == ImageDecoderLib::NVJPEG) {
    return ImplByNvJpeg(img_names, mats);
  }
  return true;
}

bool ImageDecoder::ImplByOpenCV(const std::vector<std::string> &img_names,
                                std::vector<FDMat> *mats) {
  for (size_t i = 0; i < img_names.size(); ++i) {
    cv::Mat im = cv::imread(img_names[i]);
    (*mats)[i].SetMat(im);
    (*mats)[i].layout = Layout::HWC;
    (*mats)[i].SetWidth(im.cols);
    (*mats)[i].SetHeight(im.rows);
    (*mats)[i].SetChannels(im.channels());
  }
  return true;
}

bool ImageDecoder::ImplByNvJpeg(const std::vector<std::string> &img_names,
                                std::vector<FDMat> *mats) {
#ifdef ENABLE_NVJPEG
  nvjpeg_params_.batch_size = img_names.size();
  std::vector<nvjpegImage_t> output_imgs(nvjpeg_params_.batch_size);
  std::vector<int> widths(nvjpeg_params_.batch_size);
  std::vector<int> heights(nvjpeg_params_.batch_size);
  // TODO(wangxinyu): support other output format
  nvjpeg_params_.fmt = NVJPEG_OUTPUT_BGRI;
  double total;
  nvjpeg_params_.stream = (*mats)[0].Stream();

  std::vector<FDTensor *> output_buffers;
  for (size_t i = 0; i < mats->size(); ++i) {
    FDASSERT((*mats)[i].output_cache != nullptr,
             "The output_cache of FDMat was not set.");
    output_buffers.push_back((*mats)[i].output_cache);
  }

  if (nvjpeg::process_images(img_names, nvjpeg_params_, total, output_imgs,
                             output_buffers, widths, heights)) {
    // If nvJPEG decode failed, will fallback to OpenCV,
    // e.g. png format is not supported by nvJPEG
    FDWARNING << "nvJPEG decode failed, falling back to OpenCV for this batch"
              << std::endl;
    return ImplByOpenCV(img_names, mats);
  }

  for (size_t i = 0; i < mats->size(); ++i) {
    (*mats)[i].mat_type = ProcLib::CUDA;
    (*mats)[i].layout = Layout::HWC;
    (*mats)[i].SetTensor(output_buffers[i]);
  }
#else
  FDASSERT(false, "UltraInfer didn't compile with NVJPEG.");
#endif
  return true;
}

} // namespace vision
} // namespace ultra_infer
