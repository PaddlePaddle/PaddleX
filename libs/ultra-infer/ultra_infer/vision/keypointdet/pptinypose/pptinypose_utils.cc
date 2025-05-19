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

#include "ultra_infer/vision/keypointdet/pptinypose/pptinypose_utils.h"
#define PI 3.1415926535
#define HALF_CIRCLE_DEGREE 180

namespace ultra_infer {
namespace vision {
namespace keypointdetection {

cv::Point2f Get3dPoint(const cv::Point2f &a, const cv::Point2f &b) {
  cv::Point2f direct{a.x - b.x, a.y - b.y};
  return cv::Point2f(a.x - direct.y, a.y + direct.x);
}

std::vector<float> GetDir(const float src_point_x, const float src_point_y,
                          const float rot_rad) {
  float sn = sin(rot_rad);
  float cs = cos(rot_rad);
  std::vector<float> src_result{0.0, 0.0};
  src_result[0] = src_point_x * cs - src_point_y * sn;
  src_result[1] = src_point_x * sn + src_point_y * cs;
  return src_result;
}

void AffineTransform(const float pt_x, const float pt_y, const cv::Mat &trans,
                     std::vector<float> *preds, const int p) {
  double new1[3] = {pt_x, pt_y, 1.0};
  cv::Mat new_pt(3, 1, trans.type(), new1);
  cv::Mat w = trans * new_pt;
  (*preds)[p * 3 + 1] = static_cast<float>(w.at<double>(0, 0));
  (*preds)[p * 3 + 2] = static_cast<float>(w.at<double>(1, 0));
}

void GetAffineTransform(const std::vector<float> &center,
                        const std::vector<float> &scale, const float rot,
                        const std::vector<int> &output_size, cv::Mat *trans,
                        const int inv) {
  float src_w = scale[0];
  float dst_w = static_cast<float>(output_size[0]);
  float dst_h = static_cast<float>(output_size[1]);
  float rot_rad = rot * PI / HALF_CIRCLE_DEGREE;
  std::vector<float> src_dir = GetDir(-0.5 * src_w, 0, rot_rad);
  std::vector<float> dst_dir{-0.5f * dst_w, 0.0};
  cv::Point2f srcPoint2f[3], dstPoint2f[3];
  srcPoint2f[0] = cv::Point2f(center[0], center[1]);
  srcPoint2f[1] = cv::Point2f(center[0] + src_dir[0], center[1] + src_dir[1]);
  srcPoint2f[2] = Get3dPoint(srcPoint2f[0], srcPoint2f[1]);

  dstPoint2f[0] = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
  dstPoint2f[1] =
      cv::Point2f(dst_w * 0.5 + dst_dir[0], dst_h * 0.5 + dst_dir[1]);
  dstPoint2f[2] = Get3dPoint(dstPoint2f[0], dstPoint2f[1]);
  if (inv == 0) {
    (*trans) = cv::getAffineTransform(srcPoint2f, dstPoint2f);
  } else {
    (*trans) = cv::getAffineTransform(dstPoint2f, srcPoint2f);
  }
}

void TransformPreds(std::vector<float> &coords,
                    const std::vector<float> &center,
                    const std::vector<float> &scale,
                    const std::vector<int> &output_size,
                    const std::vector<int> &dim,
                    std::vector<float> *target_coords) {
  cv::Mat trans(2, 3, CV_64FC1);
  GetAffineTransform(center, scale, 0, output_size, &trans, 1);
  for (int p = 0; p < dim[1]; ++p) {
    AffineTransform(coords[p * 2], coords[p * 2 + 1], trans, target_coords, p);
  }
}

void GetFinalPredictions(const std::vector<float> &heatmap,
                         const std::vector<int> &dim,
                         const std::vector<int64_t> &idxout,
                         const std::vector<float> &center,
                         const std::vector<float> scale,
                         std::vector<float> *preds, const bool DARK) {
  std::vector<float> coords(dim[1] * 2);

  int heatmap_height = dim[2];
  int heatmap_width = dim[3];
  for (int j = 0; j < dim[1]; ++j) {
    int index = j * dim[2] * dim[3];
    int idx = idxout[j];
    (*preds)[j * 3] = heatmap[index + idx];
    coords[j * 2] = idx % heatmap_width;
    coords[j * 2 + 1] = idx / heatmap_width;
    int px = int(coords[j * 2] + 0.5);
    int py = int(coords[j * 2 + 1] + 0.5);
    if (DARK && px > 1 && px < heatmap_width - 2) {
      utils::DarkParse(heatmap, dim, &coords, px, py, index, j);
    } else {
      if (px > 0 && px < heatmap_width - 1) {
        float diff_x = heatmap[index + py * dim[3] + px + 1] -
                       heatmap[index + py * dim[3] + px - 1];
        coords[j * 2] += diff_x > 0 ? 1 : -1 * 0.25;
      }
      if (py > 0 && py < heatmap_height - 1) {
        float diff_y = heatmap[index + (py + 1) * dim[3] + px] -
                       heatmap[index + (py - 1) * dim[3] + px];
        coords[j * 2 + 1] += diff_y > 0 ? 1 : -1 * 0.25;
      }
    }
  }
  std::vector<int> img_size{heatmap_width, heatmap_height};
  TransformPreds(coords, center, scale, img_size, dim, preds);
}

} // namespace keypointdetection
} // namespace vision
} // namespace ultra_infer
