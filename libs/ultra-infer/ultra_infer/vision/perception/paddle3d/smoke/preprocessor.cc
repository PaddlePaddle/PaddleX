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

#include "ultra_infer/vision/perception/paddle3d/smoke/preprocessor.h"

#include "ultra_infer/function/concat.h"
#include "yaml-cpp/yaml.h"

namespace ultra_infer {
namespace vision {
namespace perception {

SmokePreprocessor::SmokePreprocessor(const std::string &config_file) {
  config_file_ = config_file;
  FDASSERT(BuildPreprocessPipelineFromConfig(),
           "Failed to create Paddle3DDetPreprocessor.");
  initialized_ = true;
}

bool SmokePreprocessor::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile &e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  // read for preprocess
  processors_.push_back(std::make_shared<BGR2RGB>());

  bool has_permute = false;
  for (const auto &op : cfg["Preprocess"]) {
    std::string op_name = op["type"].as<std::string>();
    if (op_name == "NormalizeImage") {
      auto mean = op["mean"].as<std::vector<float>>();
      auto std = op["std"].as<std::vector<float>>();
      bool is_scale = true;
      if (op["is_scale"]) {
        is_scale = op["is_scale"].as<bool>();
      }
      std::string norm_type = "mean_std";
      if (op["norm_type"]) {
        norm_type = op["norm_type"].as<std::string>();
      }
      if (norm_type != "mean_std") {
        std::fill(mean.begin(), mean.end(), 0.0);
        std::fill(std.begin(), std.end(), 1.0);
      }
      processors_.push_back(std::make_shared<Normalize>(mean, std, is_scale));
    } else if (op_name == "Resize") {
      bool keep_ratio = op["keep_ratio"].as<bool>();
      auto target_size = op["target_size"].as<std::vector<int>>();
      int interp = op["interp"].as<int>();
      FDASSERT(target_size.size() == 2,
               "Require size of target_size be 2, but now it's %lu.",
               target_size.size());
      if (!keep_ratio) {
        int width = target_size[1];
        int height = target_size[0];
        processors_.push_back(
            std::make_shared<Resize>(width, height, -1.0, -1.0, interp, false));
      } else {
        int min_target_size = std::min(target_size[0], target_size[1]);
        int max_target_size = std::max(target_size[0], target_size[1]);
        std::vector<int> max_size;
        if (max_target_size > 0) {
          max_size.push_back(max_target_size);
          max_size.push_back(max_target_size);
        }
        processors_.push_back(std::make_shared<ResizeByShort>(
            min_target_size, interp, true, max_size));
      }
    } else if (op_name == "Permute") {
      // Do nothing, do permute as the last operation
      has_permute = true;
      continue;
    } else {
      FDERROR << "Unexcepted preprocess operator: " << op_name << "."
              << std::endl;
      return false;
    }
  }
  if (!disable_permute_) {
    if (has_permute) {
      // permute = cast<float> + HWC2CHW
      processors_.push_back(std::make_shared<Cast>("float"));
      processors_.push_back(std::make_shared<HWC2CHW>());
    }
  }

  // Fusion will improve performance
  FuseTransforms(&processors_);

  input_k_data_ = cfg["k_data"].as<std::vector<float>>();
  input_ratio_data_ = cfg["ratio_data"].as<std::vector<float>>();
  return true;
}

bool SmokePreprocessor::Apply(FDMatBatch *image_batch,
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
  // There are 3 outputs, image, k_data, ratio_data
  outputs->resize(3);
  int batch = static_cast<int>(image_batch->mats->size());

  // Allocate memory for k_data
  (*outputs)[2].Resize({batch, 3, 3}, FDDataType::FP32);

  // Allocate memory for ratio_data
  (*outputs)[0].Resize({batch, 2}, FDDataType::FP32);

  auto *k_data_ptr = reinterpret_cast<float *>((*outputs)[2].MutableData());

  auto *ratio_data_ptr = reinterpret_cast<float *>((*outputs)[0].MutableData());

  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat *mat = &(image_batch->mats->at(i));
    for (size_t j = 0; j < processors_.size(); ++j) {
      if (!(*(processors_[j].get()))(mat)) {
        FDERROR << "Failed to processs image:" << i << " in "
                << processors_[j]->Name() << "." << std::endl;
        return false;
      }
    }

    memcpy(k_data_ptr + i * 9, input_k_data_.data(), 9 * sizeof(float));
    memcpy(ratio_data_ptr + i * 2, input_ratio_data_.data(), 2 * sizeof(float));
  }

  FDTensor *tensor = image_batch->Tensor();
  (*outputs)[1].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);
  return true;
}

} // namespace perception
} // namespace vision
} // namespace ultra_infer
