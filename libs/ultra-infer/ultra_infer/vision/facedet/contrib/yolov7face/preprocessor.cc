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

#include "ultra_infer/vision/facedet/contrib/yolov7face/preprocessor.h"
#include "ultra_infer/function/concat.h"
#include "ultra_infer/vision/common/processors/mat.h"

namespace ultra_infer {

namespace vision {

namespace facedet {

Yolov7FacePreprocessor::Yolov7FacePreprocessor() {
  size_ = {640, 640};
  padding_color_value_ = {114.0, 114.0, 114.0};
  is_mini_pad_ = false;
  is_no_pad_ = false;
  is_scale_up_ = false;
  stride_ = 32;
  max_wh_ = 7680.0;
}

bool Yolov7FacePreprocessor::Run(
    std::vector<FDMat> *images, std::vector<FDTensor> *outputs,
    std::vector<std::map<std::string, std::array<float, 2>>> *ims_info) {
  if (images->size() == 0) {
    FDERROR << "The size of input images should be greater than 0."
            << std::endl;
    return false;
  }
  ims_info->resize(images->size());
  outputs->resize(1);
  std::vector<FDTensor> tensors(images->size());
  for (size_t i = 0; i < images->size(); i++) {
    if (!Preprocess(&(*images)[i], &tensors[i], &(*ims_info)[i])) {
      FDERROR << "Failed to preprocess input image." << std::endl;
      return false;
    }
  }

  if (tensors.size() == 1) {
    (*outputs)[0] = std::move(tensors[0]);
  } else {
    function::Concat(tensors, &((*outputs)[0]), 0);
  }
  return true;
}

bool Yolov7FacePreprocessor::Preprocess(
    FDMat *mat, FDTensor *output,
    std::map<std::string, std::array<float, 2>> *im_info) {
  // Record the shape of image and the shape of preprocessed image
  (*im_info)["input_shape"] = {static_cast<float>(mat->Height()),
                               static_cast<float>(mat->Width())};

  // yolov7-face's preprocess steps
  // 1. letterbox
  // 2. convert_and_permute(swap_rb=true)
  LetterBox(mat);
  std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  std::vector<float> beta = {0.0f, 0.0f, 0.0f};
  ConvertAndPermute::Run(mat, alpha, beta, true);

  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {static_cast<float>(mat->Height()),
                                static_cast<float>(mat->Width())};

  mat->ShareWithTensor(output);
  output->ExpandDim(0); // reshape to n, c, h, w
  return true;
}

void Yolov7FacePreprocessor::LetterBox(FDMat *mat) {
  float scale =
      std::min(size_[1] * 1.0 / mat->Height(), size_[0] * 1.0 / mat->Width());
  if (!is_scale_up_) {
    scale = std::min(scale, 1.0f);
  }

  int resize_h = int(round(mat->Height() * scale));
  int resize_w = int(round(mat->Width() * scale));

  int pad_w = size_[0] - resize_w;
  int pad_h = size_[1] - resize_h;
  if (is_mini_pad_) {
    pad_h = pad_h % stride_;
    pad_w = pad_w % stride_;
  } else if (is_no_pad_) {
    pad_h = 0;
    pad_w = 0;
    resize_h = size_[1];
    resize_w = size_[0];
  }
  Resize::Run(mat, resize_w, resize_h);

  if (pad_h > 0 || pad_w > 0) {
    float half_h = pad_h * 1.0 / 2;
    int top = int(round(half_h - 0.1));
    int bottom = int(round(half_h + 0.1));
    float half_w = pad_w * 1.0 / 2;
    int left = int(round(half_w - 0.1));
    int right = int(round(half_w + 0.1));
    Pad::Run(mat, top, bottom, left, right, padding_color_value_);
  }
}

} // namespace facedet

} // namespace vision

} // namespace ultra_infer
