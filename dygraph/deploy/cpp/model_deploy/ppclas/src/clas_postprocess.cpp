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

#include <time.h>

#include "model_deploy/ppclas/include/clas_postprocess.h"

namespace PaddleDeploy {

bool ClasPostprocess::Init(const YAML::Node& yaml_config) {
  labels_.clear();
  for (auto item : yaml_config["labels"]) {
    std::string label = item.as<std::string>();
    labels_.push_back(label);
  }
  return true;
}

bool ClasPostprocess::Run(const std::vector<DataBlob>& outputs,
                         const std::vector<ShapeInfo>& shape_infos,
                         std::vector<Result>* results, int thread_num) {
  if (outputs.size() == 0) {
    std::cerr << "empty output on ClasPostprocess" << std::endl;
    return false;
  }
  results->clear();
  int batch_size = shape_infos.size();
  results->resize(batch_size);

  const float* result_data =
        reinterpret_cast<const float*>(outputs[0].data.data());
  int total_size = std::accumulate(outputs[0].shape.begin(),
                                   outputs[0].shape.end(),
                                   1, std::multiplies<int>());
  int single_size = total_size / batch_size;

  #pragma omp parallel for num_threads(thread_num)
  for (int i = 0; i < batch_size; ++i) {
    (*results)[i].model_type = "clas";
    (*results)[i].clas_result = new ClasResult();
    const float* start_ptr = result_data + i * single_size;
    const float* end_ptr = result_data + (i + 1) * single_size;
    const float* ptr = std::max_element(start_ptr, end_ptr);
    (*results)[i].clas_result->category_id = std::distance(start_ptr, ptr);
    if ((*results)[i].clas_result->category_id >= labels_.size()) {
      std::cerr << "Compute category id is greater than labels "
                << "in your config file" << std::endl;
      std::cerr << "Compute Category ID: "
                << (*results)[i].clas_result->category_id
                << ", but length of labels is " << labels_.size()
                << std::endl;
    }
    (*results)[i].clas_result->category =
        labels_[(*results)[i].clas_result->category_id];
    (*results)[i].clas_result->score = *ptr;
  }
  return true;
}

}  //  namespace PaddleDeploy
