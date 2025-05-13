// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "ultra_infer/vision/classification/ppshitu/ppshituv2_rec_preprocessor.h"

#include "yaml-cpp/yaml.h"

namespace ultra_infer {
namespace vision {
namespace classification {

PPShiTuV2RecognizerPreprocessor::PPShiTuV2RecognizerPreprocessor(
    const std::string &config_file) {
  this->config_file_ = config_file;
  FDASSERT(BuildPreprocessPipelineFromConfig(),
           "Failed to create PPShiTuV2RecognizerPreprocessor.");
  initialized_ = true;
}

bool PPShiTuV2RecognizerPreprocessor::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile &e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }
  auto preprocess_cfg = cfg["PreProcess"]["transform_ops"];
  // Outdated:
  // We use the key 'RecPreProcess' to denote the preprocess
  // operators for PP-ShiTuV2 recognizer.
  // auto preprocess_cfg = cfg["RecPreProcess"]["transform_ops"];
  processors_.push_back(std::make_shared<BGR2RGB>());
  for (const auto &op : preprocess_cfg) {
    FDASSERT(op.IsMap(),
             "Require the transform information in yaml be Map type.");
    auto op_name = op.begin()->first.as<std::string>();
    if (op_name == "ResizeImage") {
      if (op.begin()->second["resize_short"]) {
        int target_size = op.begin()->second["resize_short"].as<int>();
        bool use_scale = false;
        int interp = 1;
        processors_.push_back(
            std::make_shared<ResizeByShort>(target_size, 1, use_scale));
      } else if (op.begin()->second["size"]) {
        int width = 0;
        int height = 0;
        if (op.begin()->second["size"].IsScalar()) {
          auto size = op.begin()->second["size"].as<int>();
          width = size;
          height = size;
        } else {
          auto size = op.begin()->second["size"].as<std::vector<int>>();
          width = size[0];
          height = size[1];
        }
        processors_.push_back(
            std::make_shared<Resize>(width, height, -1.0, -1.0, 1, false));
      } else {
        FDERROR << "Invalid params for ResizeImage for both 'size' and "
                   "'resize_short' are None"
                << std::endl;
      }

    } else if (op_name == "CropImage") {
      int width = op.begin()->second["size"].as<int>();
      int height = op.begin()->second["size"].as<int>();
      processors_.push_back(std::make_shared<CenterCrop>(width, height));
    } else if (op_name == "NormalizeImage") {
      if (!disable_normalize_) {
        auto mean = op.begin()->second["mean"].as<std::vector<float>>();
        auto std = op.begin()->second["std"].as<std::vector<float>>();
        const auto &scale_origin = op.begin()->second["scale"];
        float scale;
        if (scale_origin.as<std::string>() == "1/255") {
          scale = 1.0f / 255.0f;
        } else {
          scale = scale_origin.as<float>();
        }
        processors_.push_back(std::make_shared<Normalize>(
            mean, std, true, std::vector<float>(mean.size(), 0.0f),
            std::vector<float>(mean.size(), 1.0f / scale)));
      }
    } else if (op_name == "ToCHWImage") {
      if (!disable_permute_) {
        processors_.push_back(std::make_shared<HWC2CHW>());
      }
    } else {
      FDERROR << "Unexcepted preprocess operator: " << op_name << "."
              << std::endl;
      return false;
    }
  }

  // Fusion will improve performance
  FuseTransforms(&processors_);
  return true;
}

void PPShiTuV2RecognizerPreprocessor::DisableNormalize() {
  this->disable_normalize_ = true;
  // the DisableNormalize function will be invalid if the configuration file is
  // loaded during preprocessing
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
  }
}
void PPShiTuV2RecognizerPreprocessor::DisablePermute() {
  this->disable_permute_ = true;
  // the DisablePermute function will be invalid if the configuration file is
  // loaded during preprocessing
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
  }
}

bool PPShiTuV2RecognizerPreprocessor::Apply(FDMatBatch *image_batch,
                                            std::vector<FDTensor> *outputs) {
  if (!initialized_) {
    FDERROR << "The preprocessor is not initialized." << std::endl;
    return false;
  }
  for (size_t j = 0; j < processors_.size(); ++j) {
    image_batch->proc_lib = proc_lib_;
    if (initial_resize_on_cpu_ && j == 0 &&
        processors_[j]->Name().find("Resize") == 0) {
      image_batch->proc_lib = ProcLib::OPENCV;
    }
    if (!(*(processors_[j].get()))(image_batch)) {
      FDERROR << "Failed to processs image in " << processors_[j]->Name() << "."
              << std::endl;
      return false;
    }
  }

  outputs->resize(1);
  FDTensor *tensor = image_batch->Tensor();
  (*outputs)[0].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);
  return true;
}

} // namespace classification
} // namespace vision
} // namespace ultra_infer
