//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <inference_engine.hpp>
using namespace InferenceEngine;

namespace PaddleX {

// Abstraction of preprocessing opration class
class Transform {
 public:
  virtual void Init(const YAML::Node& item) = 0;
  virtual bool Run(cv::Mat* im) = 0;
};

class Normalize : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    mean_ = item["mean"].as<std::vector<float>>();
    std_ = item["std"].as<std::vector<float>>();
  }

  virtual bool Run(cv::Mat* im);

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
};

class ResizeByShort : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    short_size_ = item["short_size"].as<int>();
    if (item["max_size"].IsDefined()) {
      max_size_ = item["max_size"].as<int>();
    } else {
      max_size_ = -1;
    }
  };
  virtual bool Run(cv::Mat* im);

 private:
  float GenerateScale(const cv::Mat& im);
  int short_size_;
  int max_size_;
};


class CenterCrop : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    if (item["crop_size"].IsScalar()) {
      height_ = item["crop_size"].as<int>();
      width_ = item["crop_size"].as<int>();
    } else if (item["crop_size"].IsSequence()) {
      std::vector<int> crop_size = item["crop_size"].as<std::vector<int>>();
      width_ = crop_size[0];
      height_ = crop_size[1];
    }
  }
  virtual bool Run(cv::Mat* im);

 private:
  int height_;
  int width_;
};

class Transforms {
 public:
  void Init(const YAML::Node& node, bool to_rgb = true);
  std::shared_ptr<Transform> CreateTransform(const std::string& name);
  bool Run(cv::Mat* im, Blob::Ptr blob);

 private:
  std::vector<std::shared_ptr<Transform>> transforms_;
  bool to_rgb_ = true;
};

}  // namespace PaddleX
