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

#pragma once


#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "yaml-cpp/yaml.h"

#include "model_deploy/common/include/output_struct.h"


namespace PaddleDeploy {

class Transform {
 public:
  virtual void Init(const YAML::Node& item) = 0;

  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape) = 0;
  virtual std::string Name() = 0;
  virtual bool Run(cv::Mat* im) = 0;
};

class Normalize : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    std::vector<double> mean_ = item["mean"].as<std::vector<double>>();
    std::vector<double> std_ = item["std"].as<std::vector<double>>();
    bool is_scale_;
    std::vector<double> min_val_;
    std::vector<double> max_val_;
    if (item["is_scale"].IsDefined()) {
      is_scale_ = item["is_scale"];
    } else {
      is_scale_ = true;
    }
    if (item["min_val"].IsDefined()) {
      min_val_ = item["min_val"].as<std::vector<float>>();
    } else {
      min_val_ = std::vector<double>(mean_.size(), 0.);
    }
    if (item["max_val"].IsDefined()) {
      max_val_ = item["max_val"].as<std::vector<float>>();
    } else {
      max_val_ = std::vector<double>(mean_.size(), 255.);
    }

    for (auto c = 0; c < mean_.size(); c++) {
      double alpha = 1.0;
      if (is_scale_) {
        alpha /= (max_val_[c] - min_val_[c]);
      }
      alpha /= std_[c];
      double beta = -1.0 * mean_[c] / std_[c];

      alpha_.push_back(alpha);
      beta_.push_back(beta);
    }
  }
  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "Normalize"; }


 private:
  std::vector<float> alpha_;
  std::vector<float> beta_;
};

class ResizeByShort : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    target_size_ = item["target_size"].as<int>();
    if (item["interp"].IsDefined()) {
      interp_ = item["interp"].as<int>();
    } else {
      interp_ = 1;
    }
    if (item["use_scale"].IsDefined()) {
      use_scale_ = item["use_scale"].as<bool>();
    } else {
      use_scale_ = true;
    }
    if (item["max_size"].IsDefined()) {
      max_size_ = item["max_size"].as<int>();
    } else {
      max_size_ = -1;
    }
  }
  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "ResizeByShort"; }

 private:
  double GenerateScale(const int origin_w, const int origin_h);
  int target_size_;
  int max_size_;
  int interp_;
  bool use_scale_;
};

class ResizeByLong : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    target_size_ = item["target_size"].as<int>();
    if (item["interp"].IsDefined()) {
      interp_ = item["interp"].as<int>();
    } else {
      interp_ = 1;
    }
    if (item["max_size"].IsDefined()) {
      max_size_ = item["max_size"].as<int>();
    } else {
      max_size_ = -1;
    }
    if (item["stride"].IsDefined()) {
      stride_ = item["stride"].as<int>();
    } else {
      stride_ = 0;
    }
  }
  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "ResizeByLong"; }


 private:
  double GenerateScale(const int origin_w, const int origin_h);
  int target_size_;
  int max_size_;
  int interp_;
  int stride_;
};

class Resize : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    if (item["interp"].IsDefined()) {
      interp_ = item["interp"].as<int>();
    } else {
      interp_ = 1;
    }
    if (item["use_scale"].IsDefined()) {
      use_scale_ = item["use_scale"].as<bool>();
    } else {
      use_scale_ = true;
    }
    height_ = item["height"].as<int>();
    width_ = item["width"].as<int>();
    if (height_ <= 0 || width_ <= 0) {
      std::cerr << "[Resize] target_size should greater than 0" << std::endl;
      exit(-1);
    }
  }
  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "Resize"; }


 private:
  int height_;
  int width_;
  int interp_;
  bool use_scale_;
};

class BGR2RGB : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
  }
  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "BGR2RGB"; }
};

class RGB2BGR : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
  }
  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "RGB2BGR"; }
};

class Padding : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    if (item["stride"].IsDefined()) {
      stride_ = item["stride"].as<int>();
      if (stride_ < 1) {
        std::cerr << "[Padding] coarest_stride should greater than 0"
                  << std::endl;
        exit(-1);
      }
    }
    if (item["width"].IsDefined() && item["height"].IsDefined()) {
      width_ = item["width"].as<int>();
      height_ = item["height"].as<int>();
    }
    if (item["im_padding_value"].IsDefined()) {
      im_value_ = item["im_padding_value"].as<std::vector<float>>();
    } else {
      im_value_ = {0, 0, 0};
    }
  }
  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "Padding"; }

  virtual void GeneralPadding(cv::Mat* im,
                              const std::vector<float>& padding_val,
                              int padding_w, int padding_h);
  virtual void MultichannelPadding(cv::Mat* im,
                                   const std::vector<float>& padding_val,
                                   int padding_w, int padding_h);
  virtual bool Run(cv::Mat* im, int padding_w, int padding_h);

 private:
  int stride_ = -1;
  int width_ = 0;
  int height_ = 0;
  std::vector<float> im_value_;
};

class CenterCrop : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    height_ = item["width"].as<int>();
    width_ = item["height"].as<int>();
  }
  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "CenterCrop"; }


 private:
  int height_;
  int width_;
};

class Clip : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    min_val_ = item["min_val"].as<std::vector<float>>();
    max_val_ = item["max_val"].as<std::vector<float>>();
  }

  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "Clip"; }


 private:
  std::vector<float> min_val_;
  std::vector<float> max_val_;
};

class Permute : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {}
  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "Permute"; }
};

class Convert : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    dtype_ = item["dtype"].as<std::string>();
  }
  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "Convert"; }


 private:
  std::string dtype_;
};

class OcrResize : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    height_ = item["height"].as<int>();
    width_ = item["width"].as<int>();
    is_pad_ = item["is_pad"].as<bool>();
    fix_width_ = item["fix_width"].as<bool>();
    if (item["interp"].IsDefined()) {
      interp_ = item["interp"].as<int>();
    } else {
      interp_ = 1;
    }
    if (item["value"].IsDefined()) {
      std::vector<float> value = item["value"].as<std::vector<float>>();
      value_ = cv::Scalar(value[0], value[1], value[2]);
    } else {
      value_ = cv::Scalar(0, 0, 0);
    }
  }
  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "OcrResize"; }


 private:
  int GeneralWidth(int w, int h);

  int height_;
  int width_;
  int interp_;
  bool is_pad_;
  bool fix_width_;
  cv::Scalar value_;
};

class OcrTrtResize : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    height_ = item["height"].as<int>();
    width_ = item["width"].as<int>();
    if (item["interp"].IsDefined()) {
      interp_ = item["interp"].as<int>();
    } else {
      interp_ = 1;
    }
  }
  virtual bool Run(cv::Mat* im);
  virtual bool ShapeInfer(const std::vector<int>& in_shape,
                          std::vector<int>* out_shape);
  virtual std::string Name() { return "OcrTrtResize"; }


 private:
  int height_;
  int width_;
  int interp_;
};

}  // namespace PaddleDeploy
