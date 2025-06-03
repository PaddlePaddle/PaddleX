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

#include "ultra_infer/vision/utils/utils.h"

namespace ultra_infer {
namespace vision {
namespace utils {

void DarkParse(const std::vector<float> &heatmap, const std::vector<int> &dim,
               std::vector<float> *coords, const int px, const int py,
               const int index, const int ch) {
  /*DARK postprocessing, Zhang et al. Distribution-Aware Coordinate
  Representation for Human Pose Estimation (CVPR 2020).
  1) offset = - hassian.inv() * derivative
  2) dx = (heatmap[x+1] - heatmap[x-1])/2.
  3) dxx = (dx[x+1] - dx[x-1])/2.
  4) derivative = Mat([dx, dy])
  5) hassian = Mat([[dxx, dxy], [dxy, dyy]])
  */
  std::vector<float>::const_iterator first1 = heatmap.begin() + index;
  std::vector<float>::const_iterator last1 =
      heatmap.begin() + index + dim[2] * dim[3];
  std::vector<float> heatmap_ch(first1, last1);
  cv::Mat heatmap_mat = cv::Mat(heatmap_ch).reshape(0, dim[2]);
  heatmap_mat.convertTo(heatmap_mat, CV_32FC1);
  cv::GaussianBlur(heatmap_mat, heatmap_mat, cv::Size(3, 3), 0, 0);
  heatmap_mat = heatmap_mat.reshape(1, 1);
  heatmap_ch = std::vector<float>(heatmap_mat.reshape(1, 1));

  float epsilon = 1e-10;
  // sample heatmap to get values in around target location
  float xy = log(fmax(heatmap_ch[py * dim[3] + px], epsilon));
  float xr = log(fmax(heatmap_ch[py * dim[3] + px + 1], epsilon));
  float xl = log(fmax(heatmap_ch[py * dim[3] + px - 1], epsilon));

  float xr2 = log(fmax(heatmap_ch[py * dim[3] + px + 2], epsilon));
  float xl2 = log(fmax(heatmap_ch[py * dim[3] + px - 2], epsilon));
  float yu = log(fmax(heatmap_ch[(py + 1) * dim[3] + px], epsilon));
  float yd = log(fmax(heatmap_ch[(py - 1) * dim[3] + px], epsilon));
  float yu2 = log(fmax(heatmap_ch[(py + 2) * dim[3] + px], epsilon));
  float yd2 = log(fmax(heatmap_ch[(py - 2) * dim[3] + px], epsilon));
  float xryu = log(fmax(heatmap_ch[(py + 1) * dim[3] + px + 1], epsilon));
  float xryd = log(fmax(heatmap_ch[(py - 1) * dim[3] + px + 1], epsilon));
  float xlyu = log(fmax(heatmap_ch[(py + 1) * dim[3] + px - 1], epsilon));
  float xlyd = log(fmax(heatmap_ch[(py - 1) * dim[3] + px - 1], epsilon));

  // compute dx/dy and dxx/dyy with sampled values
  float dx = 0.5 * (xr - xl);
  float dy = 0.5 * (yu - yd);
  float dxx = 0.25 * (xr2 - 2 * xy + xl2);
  float dxy = 0.25 * (xryu - xryd - xlyu + xlyd);
  float dyy = 0.25 * (yu2 - 2 * xy + yd2);

  // finally get offset by derivative and hassian, which combined by dx/dy and
  // dxx/dyy
  if (dxx * dyy - dxy * dxy != 0) {
    float M[2][2] = {dxx, dxy, dxy, dyy};
    float D[2] = {dx, dy};
    cv::Mat hassian(2, 2, CV_32F, M);
    cv::Mat derivative(2, 1, CV_32F, D);
    cv::Mat offset = -hassian.inv() * derivative;
    (*coords)[ch * 2] += offset.at<float>(0, 0);
    (*coords)[ch * 2 + 1] += offset.at<float>(1, 0);
  }
}

} // namespace utils
} // namespace vision
} // namespace ultra_infer
