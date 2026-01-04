// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.  //NOLINT
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
#include "ultra_infer/vision/detection/contrib/rknpu2/utils.h"
namespace ultra_infer {
namespace vision {
namespace detection {
float Clamp(float val, int min, int max) {
  return val > min ? (val < max ? val : max) : min;
}
static float CalculateOverlap(float xmin0, float ymin0, float xmax0,
                              float ymax0, float xmin1, float ymin1,
                              float xmax1, float ymax1) {
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) +
            (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}

int NMS(int valid_count, std::vector<float> &output_locations,
        std::vector<int> &class_id, std::vector<int> &order, float threshold,
        bool class_agnostic) {
  for (int i = 0; i < valid_count; ++i) {
    if (order[i] == -1) {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < valid_count; ++j) {
      int m = order[j];
      if (m == -1) {
        continue;
      }

      if (!class_agnostic && class_id[n] != class_id[m]) {
        continue;
      }

      float xmin0 = output_locations[n * 4 + 0];
      float ymin0 = output_locations[n * 4 + 1];
      float xmax0 = output_locations[n * 4 + 0] + output_locations[n * 4 + 2];
      float ymax0 = output_locations[n * 4 + 1] + output_locations[n * 4 + 3];

      float xmin1 = output_locations[m * 4 + 0];
      float ymin1 = output_locations[m * 4 + 1];
      float xmax1 = output_locations[m * 4 + 0] + output_locations[m * 4 + 2];
      float ymax1 = output_locations[m * 4 + 1] + output_locations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1,
                                   xmax1, ymax1);

      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}
} // namespace detection
} // namespace vision
} // namespace ultra_infer
