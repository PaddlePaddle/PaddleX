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

#include "ultra_infer/vision/perception/paddle3d/caddn/preprocessor.h"

#include "ultra_infer/function/concat.h"
#include "yaml-cpp/yaml.h"

namespace ultra_infer {
namespace vision {
namespace perception {

CaddnPreprocessor::CaddnPreprocessor(const std::string &config_file) {
  config_file_ = config_file;
  FDASSERT(BuildPreprocessPipeline(),
           "Failed to create Paddle3DDetPreprocessor.");
  initialized_ = true;
}

bool CaddnPreprocessor::BuildPreprocessPipeline() {
  processors_.clear();

  // preprocess
  processors_.push_back(std::make_shared<BGR2RGB>());

  std::vector<float> alpha = {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0};
  std::vector<float> beta = {0.0, 0.0, 0.0};
  processors_.push_back(std::make_shared<Convert>(alpha, beta));

  processors_.push_back(std::make_shared<Cast>("float"));
  processors_.push_back(std::make_shared<HWC2CHW>());

  // Fusion will improve performance
  FuseTransforms(&processors_);

  return true;
}

bool CaddnPreprocessor::Apply(FDMatBatch *image_batch,
                              std::vector<float> &input_cam_data,
                              std::vector<float> &input_lidar_data,
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
  // There are 3 outputs, image, cam_data, lidar_data
  outputs->resize(3);
  int batch = static_cast<int>(image_batch->mats->size());

  // Allocate memory for cam_data
  (*outputs)[1].Resize({batch, 3, 4}, FDDataType::FP32);

  // Allocate memory for lidar_data
  (*outputs)[2].Resize({batch, 4, 4}, FDDataType::FP32);

  auto *cam_data_ptr = reinterpret_cast<float *>((*outputs)[1].MutableData());
  auto *lidar_data_ptr = reinterpret_cast<float *>((*outputs)[2].MutableData());

  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat *mat = &(image_batch->mats->at(i));
    for (size_t j = 0; j < processors_.size(); ++j) {
      if (!(*(processors_[j].get()))(mat)) {
        FDERROR << "Failed to processs image:" << i << " in "
                << processors_[j]->Name() << "." << std::endl;
        return false;
      }
    }

    memcpy(cam_data_ptr + i * 12, input_cam_data.data(), 12 * sizeof(float));
    memcpy(lidar_data_ptr + i * 16, input_lidar_data.data(),
           16 * sizeof(float));
  }

  FDTensor *tensor = image_batch->Tensor();
  (*outputs)[0].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);

  return true;
}

bool CaddnPreprocessor::Run(std::vector<FDMat> *images,
                            std::vector<float> &input_cam_data,
                            std::vector<float> &input_lidar_data,
                            std::vector<FDTensor> *outputs) {
  FDMatBatch image_batch(images);
  PreApply(&image_batch);
  bool ret = Apply(&image_batch, input_cam_data, input_lidar_data, outputs);
  PostApply();
  return ret;
}

} // namespace perception
} // namespace vision
} // namespace ultra_infer
