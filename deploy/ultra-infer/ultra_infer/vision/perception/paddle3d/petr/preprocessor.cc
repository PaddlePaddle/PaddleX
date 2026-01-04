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

#include "ultra_infer/vision/perception/paddle3d/petr/preprocessor.h"

#include <iostream>

#include "ultra_infer/function/concat.h"
#include "yaml-cpp/yaml.h"

namespace ultra_infer {
namespace vision {
namespace perception {

PetrPreprocessor::PetrPreprocessor(const std::string &config_file) {
  config_file_ = config_file;
  FDASSERT(BuildPreprocessPipelineFromConfig(),
           "Failed to create Paddle3DDetPreprocessor.");
  initialized_ = true;
}

bool PetrPreprocessor::BuildPreprocessPipelineFromConfig() {
  processors_.clear();

  processors_.push_back(std::make_shared<Resize>(800, 450));
  processors_.push_back(std::make_shared<Crop>(0, 130, 800, 320));

  std::vector<float> mean{103.530, 116.280, 123.675};
  std::vector<float> std{57.375, 57.120, 58.395};
  bool scale = false;
  processors_.push_back(std::make_shared<Normalize>(mean, std, scale));
  processors_.push_back(std::make_shared<Cast>("float"));
  processors_.push_back(std::make_shared<HWC2CHW>());

  // Fusion will improve performance
  FuseTransforms(&processors_);

  return true;
}

bool PetrPreprocessor::Apply(FDMatBatch *image_batch,
                             std::vector<FDTensor> *outputs) {
  if (image_batch->mats->empty()) {
    FDERROR << "The size of input images should be greater than 0."
            << std::endl;
    return false;
  }
  if (!initialized_) {
    FDERROR << "The preprocessor is not initialized." << std::endl;
    return false;
  }
  // There are 3 outputs, image, k_data, timestamp
  outputs->resize(3);
  int num_cams = static_cast<int>(image_batch->mats->size());

  // Allocate memory for k_data
  (*outputs)[1].Resize({1, num_cams, 4, 4}, FDDataType::FP32);

  // Allocate memory for image_data
  (*outputs)[0].Resize({1, num_cams, 3, 320, 800}, FDDataType::FP32);

  // Allocate memory for timestamp
  (*outputs)[2].Resize({1, num_cams}, FDDataType::FP32);

  auto *image_ptr = reinterpret_cast<float *>((*outputs)[0].MutableData());

  auto *k_data_ptr = reinterpret_cast<float *>((*outputs)[1].MutableData());

  auto *timestamp_ptr = reinterpret_cast<float *>((*outputs)[2].MutableData());

  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat *mat = &(image_batch->mats->at(i));
    for (size_t j = 0; j < processors_.size(); ++j) {
      if (!(*(processors_[j].get()))(mat)) {
        FDERROR << "Failed to process image:" << i << " in "
                << processors_[j]->Name() << "." << std::endl;
        return false;
      }
    }
  }

  for (int i = 0; i < num_cams / 2 * 4 * 4; ++i) {
    input_k_data_.push_back(input_k_data_[i]);
  }
  memcpy(k_data_ptr, input_k_data_.data(), num_cams * 16 * sizeof(float));

  std::vector<float> timestamp(num_cams, 0.0f);
  for (int i = num_cams / 2; i < num_cams; ++i) {
    timestamp[i] = 1.0f;
  }
  memcpy(timestamp_ptr, timestamp.data(), num_cams * sizeof(float));

  FDTensor *tensor = image_batch->Tensor(); // [num_cams,3,320,800]
  tensor->ExpandDim(0); // [num_cams,3,320,800] -> [1,num_cams,3,320,800]
  (*outputs)[0].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);
  return true;
}

} // namespace perception
} // namespace vision
} // namespace ultra_infer
