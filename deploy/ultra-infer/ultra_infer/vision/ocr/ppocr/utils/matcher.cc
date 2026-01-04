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

#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

std::vector<int> Xyxyxyxy2Xyxy(std::array<int, 8> &box) {
  int x_collect[4] = {box[0], box[2], box[4], box[6]};
  int y_collect[4] = {box[1], box[3], box[5], box[7]};
  int left = int(*std::min_element(x_collect, x_collect + 4));
  int right = int(*std::max_element(x_collect, x_collect + 4));
  int top = int(*std::min_element(y_collect, y_collect + 4));
  int bottom = int(*std::max_element(y_collect, y_collect + 4));
  std::vector<int> box1(4, 0);
  box1[0] = left;
  box1[1] = top;
  box1[2] = right;
  box1[3] = bottom;
  return box1;
}

float Dis(std::vector<int> &box1, std::vector<int> &box2) {
  float x1_1 = float(box1[0]);
  float y1_1 = float(box1[1]);
  float x2_1 = float(box1[2]);
  float y2_1 = float(box1[3]);

  float x1_2 = float(box2[0]);
  float y1_2 = float(box2[1]);
  float x2_2 = float(box2[2]);
  float y2_2 = float(box2[3]);

  float dis = std::abs(x1_2 - x1_1) + std::abs(y1_2 - y1_1) +
              std::abs(x2_2 - x2_1) + std::abs(y2_2 - y2_1);
  float dis_2 = std::abs(x1_2 - x1_1) + std::abs(y1_2 - y1_1);
  float dis_3 = std::abs(x2_2 - x2_1) + std::abs(y2_2 - y2_1);
  return dis + std::min(dis_2, dis_3);
}

float Iou(std::vector<int> &box1, std::vector<int> &box2) {
  int area1 = std::max(0, box1[2] - box1[0]) * std::max(0, box1[3] - box1[1]);
  int area2 = std::max(0, box2[2] - box2[0]) * std::max(0, box2[3] - box2[1]);

  // computing the sum_area
  int sum_area = area1 + area2;

  // find the each point of intersect rectangle
  int x1 = std::max(box1[0], box2[0]);
  int y1 = std::max(box1[1], box2[1]);
  int x2 = std::min(box1[2], box2[2]);
  int y2 = std::min(box1[3], box2[3]);

  // judge if there is an intersect
  if (y1 >= y2 || x1 >= x2) {
    return 0.0;
  } else {
    int intersect = (x2 - x1) * (y2 - y1);
    return intersect / (sum_area - intersect + 0.00000001);
  }
}

bool ComparisonDis(const std::vector<float> &dis1,
                   const std::vector<float> &dis2) {
  if (dis1[1] < dis2[1]) {
    return true;
  } else if (dis1[1] == dis2[1]) {
    return dis1[0] < dis2[0];
  } else {
    return false;
  }
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
