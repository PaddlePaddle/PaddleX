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

#include "model_deploy/ppdet/include/det_postprocess.h"

namespace PaddleDeploy {

bool DetPostProcess::Init(const YAML::Node& yaml_config) {
  labels_.clear();
  for (auto item : yaml_config["labels"]) {
    std::string label = item.as<std::string>();
    labels_.push_back(label);
  }
  version_ = yaml_config["version"].as<std::string>();
  return true;
}

bool DetPostProcess::ProcessBbox(const std::vector<DataBlob>& outputs,
                                 const std::vector<ShapeInfo>& shape_infos,
                                 std::vector<Result>* results, int thread_num) {
  const float* data = reinterpret_cast<const float*>(outputs[0].data.data());

  std::vector<int> num_bboxes_each_sample;
  if (outputs[0].lod.empty()) {
    num_bboxes_each_sample.push_back(outputs[0].data.size() / sizeof(float));
  } else {
    for (auto i = 0; i < outputs[0].lod[0].size() - 1; ++i) {
      int num = outputs[0].lod[0][i + 1] - outputs[0].lod[0][i];
      num_bboxes_each_sample.push_back(num);
    }
  }

  int idx = 0;
  for (auto i = 0; i < num_bboxes_each_sample.size(); ++i) {
    (*results)[i].model_type = "det";
    (*results)[i].det_result = new DetResult();
    for (auto j = 0; j < num_bboxes_each_sample[i]; ++j) {
      Box box;
      box.category_id = static_cast<int>(round(data[idx * 6]));
      if (box.category_id < 0) {
        std::cerr << "Compute category id is less than 0"
                  << "(Maybe no object detected)" << std::endl;
        return true;
      }
      if (box.category_id >= labels_.size()) {
        std::cerr << "Compute category id is greater than labels "
                  << "in your config file" << std::endl;
        std::cerr << "Compute Category ID: " << box.category_id
                  << ", but length of labels is " << labels_.size()
                  << std::endl;
        return false;
      }
      box.category = labels_[box.category_id];
      box.score = data[idx * 6 + 1];
      // TODO(jiangjiajun): only for RCNN and YOLO
      // lack of process for SSD and Face
      float xmin = data[idx * 6 + 2];
      float ymin = data[idx * 6 + 3];
      float xmax = data[idx * 6 + 4];
      float ymax = data[idx * 6 + 5];
      box.coordinate = {xmin, ymin, xmax - xmin, ymax - ymin};
      (*results)[i].det_result->boxes.push_back(std::move(box));
      idx += 1;
    }
  }
  return true;
}

bool DetPostProcess::Run(const std::vector<DataBlob>& outputs,
                         const std::vector<ShapeInfo>& shape_infos,
                         std::vector<Result>* results, int thread_num) {
  results->clear();
  if (outputs.size() == 0) {
    std::cerr << "empty input image on DetPostProcess" << std::endl;
    return true;
  }
  results->resize(shape_infos.size());
  if (!ProcessBbox(outputs, shape_infos, results, thread_num)) {
    std::cerr << "Error happend while process bboxes" << std::endl;
    return false;
  }
  // TODO(jiangjiajun): MaskRCNN is not implement
  //  if (outputs.size() == 2) {
  //    if ((!ProcessMask(outputs, shape_infos, results, thread_num)) {
  //      std::cerr << "Error happend while process masks" << std::endl;
  //      return false;
  //    }
  //  }
  return true;
}

}  //  namespace PaddleDeploy
