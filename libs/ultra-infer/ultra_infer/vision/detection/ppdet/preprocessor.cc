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

#include "ultra_infer/vision/detection/ppdet/preprocessor.h"

#include "ultra_infer/function/concat.h"
#include "ultra_infer/function/pad.h"
#include "yaml-cpp/yaml.h"

namespace ultra_infer {
namespace vision {
namespace detection {

PaddleDetPreprocessor::PaddleDetPreprocessor(const std::string &config_file) {
  this->config_file_ = config_file;
  FDASSERT(BuildPreprocessPipelineFromConfig(),
           "Failed to create PaddleDetPreprocessor.");
  initialized_ = true;
}

bool PaddleDetPreprocessor::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile &e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  // read for postprocess
  if (cfg["arch"].IsDefined()) {
    arch_ = cfg["arch"].as<std::string>();
  } else {
    FDERROR << "Please set model arch,"
            << "support value : SOLOv2, YOLO, SSD, RetinaNet, RCNN, Face."
            << std::endl;
    return false;
  }

  // read for preprocess
  processors_.push_back(std::make_shared<BGR2RGB>());

  bool has_permute = false;
  for (const auto &op : cfg["Preprocess"]) {
    std::string op_name = op["type"].as<std::string>();
    if (op_name == "NormalizeImage") {
      if (!disable_normalize_) {
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
      }
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
    } else if (op_name == "Pad") {
      auto size = op["size"].as<std::vector<int>>();
      auto value = op["fill_value"].as<std::vector<float>>();
      processors_.push_back(
          std::make_shared<PadToSize>(size[1], size[0], value));
    } else if (op_name == "PadStride") {
      auto stride = op["stride"].as<int>();
      processors_.push_back(
          std::make_shared<StridePad>(stride, std::vector<float>(3, 0)));
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

  return true;
}

bool PaddleDetPreprocessor::Apply(FDMatBatch *image_batch,
                                  std::vector<FDTensor> *outputs) {
  if (!initialized_) {
    FDERROR << "The preprocessor is not initialized." << std::endl;
    return false;
  }
  if (image_batch->mats->empty()) {
    FDERROR << "The size of input images should be greater than 0."
            << std::endl;
    return false;
  }

  // There are 3 outputs, image, scale_factor, im_shape
  // But im_shape is not used for all the PaddleDetection models
  // So preprocessor will output the 3 FDTensors, and how to use `im_shape`
  // is decided by the model itself
  outputs->resize(3);
  int batch = static_cast<int>(image_batch->mats->size());
  // Allocate memory for scale_factor
  (*outputs)[1].Resize({batch, 2}, FDDataType::FP32);
  // Allocate memory for im_shape
  (*outputs)[2].Resize({batch, 2}, FDDataType::FP32);
  // Record the max size for a batch of input image
  // All the tensor will pad to the max size to compose a batched tensor
  std::vector<int> max_hw({-1, -1});

  auto *scale_factor_ptr =
      reinterpret_cast<float *>((*outputs)[1].MutableData());
  auto *im_shape_ptr = reinterpret_cast<float *>((*outputs)[2].MutableData());
  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat *mat = &(image_batch->mats->at(i));
    int origin_w = mat->Width();
    int origin_h = mat->Height();
    scale_factor_ptr[2 * i] = 1.0;
    scale_factor_ptr[2 * i + 1] = 1.0;
    for (size_t j = 0; j < processors_.size(); ++j) {
      if (!(*(processors_[j].get()))(mat)) {
        FDERROR << "Failed to processs image:" << i << " in "
                << processors_[j]->Name() << "." << std::endl;
        return false;
      }
      if (processors_[j]->Name().find("Resize") != std::string::npos) {
        scale_factor_ptr[2 * i] = mat->Height() * 1.0 / origin_h;
        scale_factor_ptr[2 * i + 1] = mat->Width() * 1.0 / origin_w;
      }
    }
    if (mat->Height() > max_hw[0]) {
      max_hw[0] = mat->Height();
    }
    if (mat->Width() > max_hw[1]) {
      max_hw[1] = mat->Width();
    }
    im_shape_ptr[2 * i] = max_hw[0];
    im_shape_ptr[2 * i + 1] = max_hw[1];
  }

  // if the size of image less than max_hw, pad to max_hw
  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat *mat = &(image_batch->mats->at(i));
    if (mat->Height() < max_hw[0] || mat->Width() < max_hw[1]) {
      pad_op_->SetWidthHeight(max_hw[1], max_hw[0]);
      (*pad_op_)(mat);
    }
  }

  // Get the NCHW tensor
  FDTensor *tensor = image_batch->Tensor();
  (*outputs)[0].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);

  return true;
}

void PaddleDetPreprocessor::DisableNormalize() {
  this->disable_normalize_ = true;
  // the DisableNormalize function will be invalid if the configuration file is
  // loaded during preprocessing
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
  }
}

void PaddleDetPreprocessor::DisablePermute() {
  this->disable_permute_ = true;
  // the DisablePermute function will be invalid if the configuration file is
  // loaded during preprocessing
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
  }
}
} // namespace detection
} // namespace vision
} // namespace ultra_infer
