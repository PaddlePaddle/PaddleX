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
#include "ultra_infer/vision/segmentation/ppseg/preprocessor.h"

#include "ultra_infer/function/concat.h"
#include "yaml-cpp/yaml.h"

namespace ultra_infer {
namespace vision {
namespace segmentation {

PaddleSegPreprocessor::PaddleSegPreprocessor(const std::string &config_file) {
  this->config_file_ = config_file;
  FDASSERT(BuildPreprocessPipelineFromConfig(),
           "Failed to create PaddleSegPreprocessor.");
  initialized_ = true;
}

bool PaddleSegPreprocessor::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  processors_.push_back(std::make_shared<BGR2RGB>());
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile &e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  if (cfg["Deploy"]["transforms"]) {
    auto preprocess_cfg = cfg["Deploy"]["transforms"];
    for (const auto &op : preprocess_cfg) {
      FDASSERT(op.IsMap(),
               "Require the transform information in yaml be Map type.");
      if (op["type"].as<std::string>() == "Normalize") {
        if (!disable_normalize_) {
          std::vector<float> mean = {0.5, 0.5, 0.5};
          std::vector<float> std = {0.5, 0.5, 0.5};
          if (op["mean"]) {
            mean = op["mean"].as<std::vector<float>>();
          }
          if (op["std"]) {
            std = op["std"].as<std::vector<float>>();
          }
          processors_.push_back(std::make_shared<Normalize>(mean, std));
        }
      } else if (op["type"].as<std::string>() == "Resize") {
        is_contain_resize_op_ = true;
        const auto &target_size = op["target_size"];
        int resize_width = target_size[0].as<int>();
        int resize_height = target_size[1].as<int>();
        processors_.push_back(
            std::make_shared<Resize>(resize_width, resize_height));
      } else {
        std::string op_name = op["type"].as<std::string>();
        FDERROR << "Unexcepted preprocess operator: " << op_name << "."
                << std::endl;
        return false;
      }
    }
  }
  if (cfg["Deploy"]["input_shape"]) {
    auto input_shape = cfg["Deploy"]["input_shape"];
    int input_height = input_shape[2].as<int>();
    int input_width = input_shape[3].as<int>();
    if (input_height != -1 && input_width != -1 && !is_contain_resize_op_) {
      is_contain_resize_op_ = true;
      processors_.insert(processors_.begin(),
                         std::make_shared<Resize>(input_width, input_height));
    }
  }
  if (!disable_permute_) {
    processors_.push_back(std::make_shared<HWC2CHW>());
  }

  // Fusion will improve performance
  FuseTransforms(&processors_);
  return true;
}

bool PaddleSegPreprocessor::Apply(FDMatBatch *image_batch,
                                  std::vector<FDTensor> *outputs) {
  std::vector<FDMat> *images = image_batch->mats;
  if (!initialized_) {
    FDERROR << "The preprocessor is not initialized." << std::endl;
    return false;
  }
  if (images->size() == 0) {
    FDERROR << "The size of input images should be greater than 0."
            << std::endl;
    return false;
  }
  std::vector<std::array<int, 2>> shape_info;
  for (const auto &image : *images) {
    shape_info.push_back(
        {static_cast<int>(image.Height()), static_cast<int>(image.Width())});
  }
  (*imgs_info_)["shape_info"] = shape_info;
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (processors_[i]->Name() == "Resize") {
      auto processor = dynamic_cast<Resize *>(processors_[i].get());
      int resize_width = -1;
      int resize_height = -1;
      std::tie(resize_width, resize_height) = processor->GetWidthAndHeight();
      if (is_vertical_screen_ && (resize_width > resize_height)) {
        if (!(processor->SetWidthAndHeight(resize_height, resize_width))) {
          FDERROR << "Failed to set width and height of "
                  << processors_[i]->Name() << " processor." << std::endl;
        }
      }
      break;
    }
  }
  size_t img_num = images->size();
  // Batch preprocess : resize all images to the largest image shape in batch
  if (!is_contain_resize_op_ && img_num > 1) {
    int max_width = 0;
    int max_height = 0;
    for (size_t i = 0; i < img_num; ++i) {
      max_width = std::max(max_width, ((*images)[i]).Width());
      max_height = std::max(max_height, ((*images)[i]).Height());
    }
    pre_resize_op_->SetWidthAndHeight(max_width, max_height);
    for (size_t i = 0; i < img_num; ++i) {
      if (!(*pre_resize_op_)(&(*images)[i])) {
        FDERROR << "Failed to batch resize max_width and max_height"
                << std::endl;
      }
    }
  }
  for (size_t i = 0; i < img_num; ++i) {
    for (size_t j = 0; j < processors_.size(); ++j) {
      if (!(*(processors_[j].get()))(&((*images)[i]))) {
        FDERROR << "Failed to process image data in " << processors_[i]->Name()
                << "." << std::endl;
        return false;
      }
    }
  }
  outputs->resize(1);
  FDTensor *tensor = image_batch->Tensor();
  (*outputs)[0].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);
  return true;
}

void PaddleSegPreprocessor::DisableNormalize() {
  this->disable_normalize_ = true;
  // the DisableNormalize function will be invalid if the configuration file is
  // loaded during preprocessing
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
  }
}
void PaddleSegPreprocessor::DisablePermute() {
  this->disable_permute_ = true;
  // the DisablePermute function will be invalid if the configuration file is
  // loaded during preprocessing
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
  }
}
} // namespace segmentation
} // namespace vision
} // namespace ultra_infer
