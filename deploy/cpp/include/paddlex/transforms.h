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

namespace PaddleX {

/*
 * @brief
 * This class represents object for storing all preprocessed data
 * */
class ImageBlob {
 public:
  // Original image height and width
  std::vector<int> ori_im_size_ = std::vector<int>(2);
  // Newest image height and width after process
  std::vector<int> new_im_size_ = std::vector<int>(2);
  // Image height and width before resize
  std::vector<std::vector<int>> im_size_before_resize_;
  // Reshape order
  std::vector<std::string> reshape_order_;
  // Resize scale
  float scale = 1.0;
  // Buffer for image data after preprocessing
  std::vector<float> im_data_;

  void clear() {
    im_size_before_resize_.clear();
    reshape_order_.clear();
    im_data_.clear();
  }
};

/*
 * @brief
 * Abstraction of preprocessing operation class
 * */
class Transform {
 public:
  virtual void Init(const YAML::Node& item) = 0;
  /*
   * @brief
   * This method executes preprocessing operation on image matrix,
   * result will be returned at second parameter.
   * @param im: single image matrix to be preprocessed
   * @param data: the raw data of single image matrix after preprocessed
   * @return true if transform successfully
   * */
  virtual bool Run(cv::Mat* im, ImageBlob* data) = 0;
};

/*
 * @brief
 * This class execute normalization operation on image matrix
 * */
class Normalize : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    mean_ = item["mean"].as<std::vector<float>>();
    std_ = item["std"].as<std::vector<float>>();
  }

  virtual bool Run(cv::Mat* im, ImageBlob* data);

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
};

/*
 * @brief
 * This class execute resize by short operation on image matrix. At first, it resizes
 * the short side of image matrix to specified length. Accordingly, the long side
 * will be resized in the same proportion. If new length of long side exceeds max
 * size, the long size will be resized to max size, and the short size will be
 * resized in the same proportion
 * */
class ResizeByShort : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    short_size_ = item["short_size"].as<int>();
    if (item["max_size"].IsDefined()) {
      max_size_ = item["max_size"].as<int>();
    } else {
      max_size_ = -1;
    }
  }
  virtual bool Run(cv::Mat* im, ImageBlob* data);

 private:
  float GenerateScale(const cv::Mat& im);
  int short_size_;
  int max_size_;
};

/*
 * @brief
 * This class execute resize by long operation on image matrix. At first, it resizes
 * the long side of image matrix to specified length. Accordingly, the short side
 * will be resized in the same proportion.
 * */
class ResizeByLong : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    long_size_ = item["long_size"].as<int>();
  }
  virtual bool Run(cv::Mat* im, ImageBlob* data);

 private:
  int long_size_;
};

/*
 * @brief
 * This class execute resize operation on image matrix. It resizes width and height
 * to specified length.
 * */
class Resize : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    if (item["interp"].IsDefined()) {
      interp_ = item["interp"].as<std::string>();
    }
    if (item["target_size"].IsScalar()) {
      height_ = item["target_size"].as<int>();
      width_ = item["target_size"].as<int>();
    } else if (item["target_size"].IsSequence()) {
      std::vector<int> target_size = item["target_size"].as<std::vector<int>>();
      width_ = target_size[0];
      height_ = target_size[1];
    }
    if (height_ <= 0 || width_ <= 0) {
      std::cerr << "[Resize] target_size should greater than 0" << std::endl;
      exit(-1);
    }
  }
  virtual bool Run(cv::Mat* im, ImageBlob* data);

 private:
  int height_;
  int width_;
  std::string interp_;
};

/*
 * @brief
 * This class execute center crop operation on image matrix. It crops the center
 * of image matrix accroding to specified size.
 * */
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
  virtual bool Run(cv::Mat* im, ImageBlob* data);

 private:
  int height_;
  int width_;
};

/*
 * @brief
 * This class execute padding operation on image matrix. It makes border on edge
 * of image matrix.
 * */
class Padding : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    if (item["coarsest_stride"].IsDefined()) {
      coarsest_stride_ = item["coarsest_stride"].as<int>();
      if (coarsest_stride_ < 1) {
        std::cerr << "[Padding] coarest_stride should greater than 0"
                  << std::endl;
        exit(-1);
      }
    }
    if (item["target_size"].IsDefined()) {
      if (item["target_size"].IsScalar()) {
        width_ = item["target_size"].as<int>();
        height_ = item["target_size"].as<int>();
      } else if (item["target_size"].IsSequence()) {
        width_ = item["target_size"].as<std::vector<int>>()[0];
        height_ = item["target_size"].as<std::vector<int>>()[1];
      }
    }
    if (item["im_padding_value"].IsDefined()) {
      im_value_ = item["im_padding_value"].as<std::vector<float>>();
    }
    else {
      im_value_ = {0, 0, 0};
    }
  }
  virtual bool Run(cv::Mat* im, ImageBlob* data);

 private:
  int coarsest_stride_ = -1;
  int width_ = 0;
  int height_ = 0;
  std::vector<float> im_value_;
};
/*
 * @brief
 * This class is transform operations manager. It stores all neccessary
 * transform operations and run them in correct order.
 * */
class Transforms {
 public:
  void Init(const YAML::Node& node, bool to_rgb = true);
  std::shared_ptr<Transform> CreateTransform(const std::string& name);
  bool Run(cv::Mat* im, ImageBlob* data);

 private:
  std::vector<std::shared_ptr<Transform>> transforms_;
  bool to_rgb_ = true;
};

}  // namespace PaddleX
