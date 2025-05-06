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

#include "ultra_infer/vision/common/processors/convert_and_permute.h"

namespace ultra_infer {
namespace vision {

ConvertAndPermute::ConvertAndPermute(const std::vector<float> &alpha,
                                     const std::vector<float> &beta,
                                     bool swap_rb) {
  FDASSERT(alpha.size() == beta.size(), "ConvertAndPermute: requires the size "
                                        "of alpha equal to the size of beta.");
  FDASSERT(alpha.size() > 0 && beta.size() > 0,
           "ConvertAndPermute: requires the size of alpha and beta > 0.");
  alpha_.assign(alpha.begin(), alpha.end());
  beta_.assign(beta.begin(), beta.end());
  swap_rb_ = swap_rb;
}

bool ConvertAndPermute::ImplByOpenCV(FDMat *mat) {
  cv::Mat *im = mat->GetOpenCVMat();
  int origin_w = im->cols;
  int origin_h = im->rows;
  std::vector<cv::Mat> split_im;
  cv::split(*im, split_im);
  if (swap_rb_)
    std::swap(split_im[0], split_im[2]);
  for (int c = 0; c < im->channels(); c++) {
    split_im[c].convertTo(split_im[c], CV_32FC1, alpha_[c], beta_[c]);
  }
  cv::Mat res(origin_h, origin_w, CV_32FC(im->channels()));
  for (int i = 0; i < im->channels(); ++i) {
    cv::extractChannel(split_im[i],
                       cv::Mat(origin_h, origin_w, CV_32FC1,
                               res.ptr() + i * origin_h * origin_w * 4),
                       0);
  }

  mat->SetMat(res);
  mat->layout = Layout::CHW;
  return true;
}

#ifdef ENABLE_FLYCV
bool ConvertAndPermute::ImplByFlyCV(FDMat *mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "Only supports input with HWC layout." << std::endl;
    return false;
  }
  fcv::Mat *im = mat->GetFlyCVMat();
  if (im->channels() != 3) {
    FDERROR << "Only supports 3-channels image in FlyCV, but now it's "
            << im->channels() << "." << std::endl;
    return false;
  }
  std::vector<float> mean(3, 0);
  std::vector<float> std(3, 0);
  for (size_t i = 0; i < 3; ++i) {
    std[i] = 1.0 / alpha_[i];
    mean[i] = -1 * beta_[i] * std[i];
  }

  std::vector<uint32_t> channel_reorder_index = {0, 1, 2};
  if (swap_rb_)
    std::swap(channel_reorder_index[0], channel_reorder_index[2]);

  fcv::Mat new_im;
  fcv::normalize_to_submean_to_reorder(*im, mean, std, channel_reorder_index,
                                       new_im, false);
  mat->SetMat(new_im);
  mat->layout = Layout::CHW;
  return true;
}
#endif

bool ConvertAndPermute::Run(FDMat *mat, const std::vector<float> &alpha,
                            const std::vector<float> &beta, bool swap_rb,
                            ProcLib lib) {
  auto n = ConvertAndPermute(alpha, beta, swap_rb);
  return n(mat, lib);
}

} // namespace vision
} // namespace ultra_infer
