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

#include "ultra_infer/vision/facedet/ppdet/blazeface/preprocessor.h"
#include "ultra_infer/function/concat.h"
#include "ultra_infer/function/pad.h"
#include "ultra_infer/vision/common/processors/mat.h"
#include "yaml-cpp/yaml.h"

namespace ultra_infer {

namespace vision {

namespace facedet {

BlazeFacePreprocessor::BlazeFacePreprocessor(const std::string &config_file) {
  is_scale_ = false;
  normalize_mean_ = {123, 117, 104};
  normalize_std_ = {127.502231, 127.502231, 127.502231};
  this->config_file_ = config_file;
  FDASSERT(BuildPreprocessPipelineFromConfig(),
           "Failed to create PaddleDetPreprocessor.");
}

bool BlazeFacePreprocessor::Run(
    std::vector<FDMat> *images, std::vector<FDTensor> *outputs,
    std::vector<std::map<std::string, std::array<float, 2>>> *ims_info) {
  if (images->size() == 0) {
    FDERROR << "The size of input images should be greater than 0."
            << std::endl;
    return false;
  }
  ims_info->resize(images->size());
  outputs->resize(3);
  int batch = static_cast<int>(images->size());
  // Allocate memory for scale_factor
  (*outputs)[1].Resize({batch, 2}, FDDataType::FP32);
  // Allocate memory for im_shape
  (*outputs)[2].Resize({batch, 2}, FDDataType::FP32);

  std::vector<int> max_hw({-1, -1});

  auto *scale_factor_ptr =
      reinterpret_cast<float *>((*outputs)[1].MutableData());
  auto *im_shape_ptr = reinterpret_cast<float *>((*outputs)[2].MutableData());

  // Concat all the preprocessed data to a batch tensor
  std::vector<FDTensor> im_tensors(images->size());

  for (size_t i = 0; i < images->size(); ++i) {
    int origin_w = (*images)[i].Width();
    int origin_h = (*images)[i].Height();
    scale_factor_ptr[2 * i] = 1.0;
    scale_factor_ptr[2 * i + 1] = 1.0;

    for (size_t j = 0; j < processors_.size(); ++j) {
      if (!(*(processors_[j].get()))(&((*images)[i]))) {
        FDERROR << "Failed to process image:" << i << " in "
                << processors_[i]->Name() << "." << std::endl;
        return false;
      }
      if (processors_[j]->Name().find("Resize") != std::string::npos) {
        scale_factor_ptr[2 * i] = (*images)[i].Height() * 1.0 / origin_h;
        scale_factor_ptr[2 * i + 1] = (*images)[i].Width() * 1.0 / origin_w;
      }
    }

    if ((*images)[i].Height() > max_hw[0]) {
      max_hw[0] = (*images)[i].Height();
    }
    if ((*images)[i].Width() > max_hw[1]) {
      max_hw[1] = (*images)[i].Width();
    }
    im_shape_ptr[2 * i] = max_hw[0];
    im_shape_ptr[2 * i + 1] = max_hw[1];

    if ((*images)[i].Height() < max_hw[0] || (*images)[i].Width() < max_hw[1]) {
      // if the size of image less than max_hw, pad to max_hw
      FDTensor tensor;
      (*images)[i].ShareWithTensor(&tensor);
      function::Pad(tensor, &(im_tensors[i]),
                    {0, 0, max_hw[0] - (*images)[i].Height(),
                     max_hw[1] - (*images)[i].Width()},
                    0);
    } else {
      // No need pad
      (*images)[i].ShareWithTensor(&(im_tensors[i]));
    }
    // Reshape to 1xCxHxW
    im_tensors[i].ExpandDim(0);
  }

  if (im_tensors.size() == 1) {
    // If there's only 1 input, no need to concat
    // skip memory copy
    (*outputs)[0] = std::move(im_tensors[0]);
  } else {
    // Else concat the im tensor for each input image
    // compose a batched input tensor
    function::Concat(im_tensors, &((*outputs)[0]), 0);
  }

  return true;
}

bool BlazeFacePreprocessor::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile &e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

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
    } else if (op_name == "Pad") {
      auto size = op["size"].as<std::vector<int>>();
      auto value = op["fill_value"].as<std::vector<float>>();
      processors_.push_back(std::make_shared<Cast>("float"));
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

  if (has_permute) {
    // permute = cast<float> + HWC2CHW
    processors_.push_back(std::make_shared<Cast>("float"));
    processors_.push_back(std::make_shared<HWC2CHW>());
  }

  // Fusion will improve performance
  FuseTransforms(&processors_);

  return true;
}

} // namespace facedet

} // namespace vision

} // namespace ultra_infer
