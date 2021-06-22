// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "model_deploy/utils/include/bbox_utils.h"

namespace PaddleDeploy {

bool FilterBbox(const std::vector<Result> &results,
                const float &score_thresh,
                std::vector<Result>* filter_results) {
  for (auto i = 0; i < results.size(); ++i) {
    if ("det" != results[i].model_type) {
       std::cerr << "FilterBbox can be only done on results from a det model, "
                 << "but the received results are from a "
                 << results[i].model_type << " model." << std::endl;
       return false;
    }
  }

  for (auto i = 0; i < results.size(); ++i) {
    Result result;
    result.model_type = "det";
    result.det_result = new DetResult();
    std::vector<Box> boxes = results[i].det_result->boxes;
    for (auto j = 0; j < boxes.size(); ++j) {
      if (boxes[j].score >= score_thresh) {
        Box box;
        box.category_id = boxes[j].category_id;
        box.category = boxes[j].category;
        box.score = boxes[j].score;
        box.coordinate.assign(boxes[j].coordinate.begin(),
                              boxes[j].coordinate.end());
        box.mask.data.assign(boxes[j].mask.data.begin(),
                             boxes[j].mask.data.end());
        box.mask.shape.assign(boxes[j].mask.shape.begin(),
                              boxes[j].mask.shape.end());
        result.det_result->boxes.push_back(std::move(box));
      }
    }
    result.det_result->mask_resolution = results[i].det_result->mask_resolution;
    filter_results->push_back(std::move(result));
  }
  return true;
}

}  // namespace PaddleDeploy
