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

#include "model_deploy/ppseg/include/seg_preprocess.h"

namespace PaddleDeploy {

bool SegPreProcess::Init(const YAML::Node& yaml_config) {
  if (!BuildTransform(yaml_config)) {
    return false;
  }
  return true;
}

bool SegPreProcess::PrepareInputs(const std::vector<ShapeInfo>& shape_infos,
                                  std::vector<cv::Mat>* imgs,
                                  std::vector<DataBlob>* inputs,
                                  int thread_num) {
  inputs->clear();
  if (!PreprocessImages(shape_infos, imgs, thread_num = thread_num)) {
    std::cerr << "Error happend while execute function "
              << "SegPreProcess::Run" << std::endl;
    return false;
  }

  DataBlob im("x");
  int batch = imgs->size();
  int w = shape_infos[0].shapes.back()[0];
  int h = shape_infos[0].shapes.back()[1];

  im.Resize({batch, 3, h, w}, FLOAT32);
  int sample_shape = 3 * h * w;
  #pragma omp parallel for num_threads(thread_num)
  for (auto i = 0; i < batch; ++i) {
    memcpy(im.data.data() + i * sample_shape * sizeof(float), (*imgs)[i].data,
            sample_shape * sizeof(float));
  }
  inputs->clear();
  inputs->push_back(std::move(im));
  return true;
}

bool SegPreProcess::Run(std::vector<cv::Mat>* imgs,
                        std::vector<DataBlob>* inputs,
                        std::vector<ShapeInfo>* shape_infos, int thread_num) {
  if (!ShapeInfer(*imgs, shape_infos, thread_num)) {
    std::cerr << "ShapeInfer failed while call SegPreProcess::Run" << std::endl;
    return false;
  }
  if (!PrepareInputs(*shape_infos, imgs, inputs, thread_num)) {
    std::cerr << "PrepareInputs failed while call "
              << "SegPreProcess::PrepareInputs" << std::endl;
    return false;
  }
  return true;
}

}  //  namespace PaddleDeploy
