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

#include "model_deploy/common/include/base_preprocess.h"

#include <omp.h>

namespace PaddleDeploy {

bool BasePreprocess::BuildTransform(const YAML::Node& yaml_config) {
  transforms_.clear();
  YAML::Node transforms_node = yaml_config["transforms"];
  for (YAML::const_iterator it = transforms_node.begin();
       it != transforms_node.end(); ++it) {
    std::string name = it->first.as<std::string>();
    std::shared_ptr<Transform> transform = CreateTransform(name);
    if (!transform) {
      std::cerr << "Failed to create " << name << " on Preprocess" << std::endl;
      return false;
    }
    transform->Init(it->second);
    transforms_.push_back(transform);
  }
  return true;
}

bool BasePreprocess::ShapeInfer(const std::vector<cv::Mat>& imgs,
                                std::vector<ShapeInfo>* shape_infos,
                                int thread_num) {
  int batch_size = imgs.size();
  thread_num = std::min(thread_num, batch_size);
  shape_infos->resize(batch_size);

  std::vector<int> success(batch_size, 1);
  #pragma omp parallel for num_threads(thread_num)
  for (auto i = 0; i < batch_size; ++i) {
    int h = imgs[i].rows;
    int w = imgs[i].cols;
    (*shape_infos)[i].Insert("Origin", w, h);
    for (auto j = 0; j < transforms_.size(); ++j) {
      std::vector<int> out_shape;
      if (!transforms_[j]->ShapeInfer((*shape_infos)[i].shapes[j],
                                      &out_shape)) {
        std::cerr << "Run transforms ShapeInfer failed!" << std::endl;
        success[i] = 0;
        continue;
      }
      (*shape_infos)[i].Insert(transforms_[j]->Name(), out_shape[0],
                               out_shape[1]);
    }
  }
  if (std::accumulate(success.begin(), success.end(), 0) < batch_size) {
    return false;
  }

  // get max shape
  int max_w = 0;
  int max_h = 0;
  for (auto i = 0; i < shape_infos->size(); ++i) {
    if ((*shape_infos)[i].shapes.back()[0] > max_w) {
      max_w = (*shape_infos)[i].shapes[transforms_.size()][0];
    }
    if ((*shape_infos)[i].shapes.back()[1] > max_h) {
      max_h = (*shape_infos)[i].shapes[transforms_.size()][1];
    }
  }
  for (auto i = 0; i < shape_infos->size(); ++i) {
    (*shape_infos)[i].Insert("Padding", max_w, max_h);
  }
  return true;
}

bool BasePreprocess::PreprocessImages(const std::vector<ShapeInfo>& shape_infos,
                                      std::vector<cv::Mat>* imgs,
                                      int thread_num) {
  int batch_size = imgs->size();
  thread_num = std::min(thread_num, batch_size);

  int max_w = shape_infos[0].shapes.back()[0];
  int max_h = shape_infos[0].shapes.back()[1];

  std::vector<int> success(batch_size, 1);

  #pragma omp parallel for num_threads(thread_num)
  for (auto i = 0; i < batch_size; ++i) {
    bool to_chw = false;
    for (auto j = 0; j < transforms_.size(); ++j) {
      // Permute will put to the last step to apply
      if (transforms_[j]->Name() == "Permute") {
        to_chw = true;
        continue;
      }
      if (!transforms_[j]->Run(&(*imgs)[i])) {
        std::cerr << "Run transforms to image failed!" << std::endl;
        success[i] = 0;
        continue;
      }
    }
    if (!batch_padding_.Run(&(*imgs)[i], max_w, max_h)) {
      std::cerr << "Run BatchPadding to image failed!" << std::endl;
      success[i] = 0;
    }

    // apply permute hwc->chw
    if (to_chw) {
      if (!permute_.Run(&(*imgs)[i])) {
        std::cerr << "Run Permute to image failed!" << std::endl;
        success[i] == 0;
      }
    }
  }

  if (std::accumulate(success.begin(), success.end(), 0) < batch_size) {
    return false;
  }
  return true;
}

std::shared_ptr<Transform> BasePreprocess::CreateTransform(
    const std::string& transform_name) {
  if (transform_name == "Normalize") {
    return std::make_shared<Normalize>();
  } else if (transform_name == "ResizeByShort") {
    return std::make_shared<ResizeByShort>();
  } else if (transform_name == "ResizeByLong") {
    return std::make_shared<ResizeByLong>();
  } else if (transform_name == "CenterCrop") {
    return std::make_shared<CenterCrop>();
  } else if (transform_name == "Permute") {
    return std::make_shared<Permute>();
  } else if (transform_name == "Resize") {
    return std::make_shared<Resize>();
  } else if (transform_name == "Padding") {
    return std::make_shared<Padding>();
  } else if (transform_name == "Clip") {
    return std::make_shared<Clip>();
  } else if (transform_name == "RGB2BGR") {
    return std::make_shared<RGB2BGR>();
  } else if (transform_name == "BGR2RGB") {
    return std::make_shared<BGR2RGB>();
  } else if (transform_name == "Convert") {
    return std::make_shared<Convert>();
  } else if (transform_name == "OcrResize") {
    return std::make_shared<OcrResize>();
  } else if (transform_name == "OcrTrtResize") {
    return std::make_shared<OcrTrtResize>();
  } else {
    std::cerr << "There's unexpected transform(name='" << transform_name
              << "')." << std::endl;
    return nullptr;
  }
}

}  // namespace PaddleDeploy
