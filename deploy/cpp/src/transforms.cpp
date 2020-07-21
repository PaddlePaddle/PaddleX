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

#include <iostream>
#include <string>
#include <vector>

#include "include/paddlex/transforms.h"

namespace PaddleX {

std::map<std::string, int> interpolations = {{"LINEAR", cv::INTER_LINEAR},
                                             {"NEAREST", cv::INTER_NEAREST},
                                             {"AREA", cv::INTER_AREA},
                                             {"CUBIC", cv::INTER_CUBIC},
                                             {"LANCZOS4", cv::INTER_LANCZOS4}};

bool Normalize::Run(cv::Mat* im, ImageBlob* data) {
  for (int h = 0; h < im->rows; h++) {
    for (int w = 0; w < im->cols; w++) {
      im->at<cv::Vec3f>(h, w)[0] =
          (im->at<cv::Vec3f>(h, w)[0] / 255.0 - mean_[0]) / std_[0];
      im->at<cv::Vec3f>(h, w)[1] =
          (im->at<cv::Vec3f>(h, w)[1] / 255.0 - mean_[1]) / std_[1];
      im->at<cv::Vec3f>(h, w)[2] =
          (im->at<cv::Vec3f>(h, w)[2] / 255.0 - mean_[2]) / std_[2];
    }
  }
  return true;
}

float ResizeByShort::GenerateScale(const cv::Mat& im) {
  int origin_w = im.cols;
  int origin_h = im.rows;
  int im_size_max = std::max(origin_w, origin_h);
  int im_size_min = std::min(origin_w, origin_h);
  float scale =
      static_cast<float>(short_size_) / static_cast<float>(im_size_min);
  if (max_size_ > 0) {
    if (round(scale * im_size_max) > max_size_) {
      scale = static_cast<float>(max_size_) / static_cast<float>(im_size_max);
    }
  }
  return scale;
}

bool ResizeByShort::Run(cv::Mat* im, ImageBlob* data) {
  data->im_size_before_resize_.push_back({im->rows, im->cols});
  data->reshape_order_.push_back("resize");

  float scale = GenerateScale(*im);
  int width = static_cast<int>(scale * im->cols);
  int height = static_cast<int>(scale * im->rows);
  cv::resize(*im, *im, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

  data->new_im_size_[0] = im->rows;
  data->new_im_size_[1] = im->cols;
  data->scale = scale;
  return true;
}

bool CenterCrop::Run(cv::Mat* im, ImageBlob* data) {
  int height = static_cast<int>(im->rows);
  int width = static_cast<int>(im->cols);
  if (height < height_ || width < width_) {
    std::cerr << "[CenterCrop] Image size less than crop size" << std::endl;
    return false;
  }
  int offset_x = static_cast<int>((width - width_) / 2);
  int offset_y = static_cast<int>((height - height_) / 2);
  cv::Rect crop_roi(offset_x, offset_y, width_, height_);
  *im = (*im)(crop_roi);
  data->new_im_size_[0] = im->rows;
  data->new_im_size_[1] = im->cols;
  return true;
}

bool Padding::Run(cv::Mat* im, ImageBlob* data) {
  data->im_size_before_resize_.push_back({im->rows, im->cols});
  data->reshape_order_.push_back("padding");

  int padding_w = 0;
  int padding_h = 0;
  if (width_ > 1 & height_ > 1) {
    padding_w = width_ - im->cols;
    padding_h = height_ - im->rows;
  } else if (coarsest_stride_ >= 1) {
    int h = im->rows;
    int w = im->cols;
    padding_h =
        ceil(h * 1.0 / coarsest_stride_) * coarsest_stride_ - im->rows;
    padding_w =
        ceil(w * 1.0 / coarsest_stride_) * coarsest_stride_ - im->cols;
  }

  if (padding_h < 0 || padding_w < 0) {
    std::cerr << "[Padding] Computed padding_h=" << padding_h
              << ", padding_w=" << padding_w
              << ", but they should be greater than 0." << std::endl;
    return false;
  }
  cv::Scalar value = cv::Scalar(im_value_[0], im_value_[1], im_value_[2]);
  cv::copyMakeBorder(
      *im, *im, 0, padding_h, 0, padding_w, cv::BORDER_CONSTANT, value);
  data->new_im_size_[0] = im->rows;
  data->new_im_size_[1] = im->cols;
  return true;
}

bool ResizeByLong::Run(cv::Mat* im, ImageBlob* data) {
  if (long_size_ <= 0) {
    std::cerr << "[ResizeByLong] long_size should be greater than 0"
              << std::endl;
    return false;
  }
  data->im_size_before_resize_.push_back({im->rows, im->cols});
  data->reshape_order_.push_back("resize");
  int origin_w = im->cols;
  int origin_h = im->rows;

  int im_size_max = std::max(origin_w, origin_h);
  float scale =
      static_cast<float>(long_size_) / static_cast<float>(im_size_max);
  cv::resize(*im, *im, cv::Size(), scale, scale, cv::INTER_NEAREST);
  data->new_im_size_[0] = im->rows;
  data->new_im_size_[1] = im->cols;
  data->scale = scale;
  return true;
}

bool Resize::Run(cv::Mat* im, ImageBlob* data) {
  if (width_ <= 0 || height_ <= 0) {
    std::cerr << "[Resize] width and height should be greater than 0"
              << std::endl;
    return false;
  }
  if (interpolations.count(interp_) <= 0) {
    std::cerr << "[Resize] Invalid interpolation method: '" << interp_ << "'"
              << std::endl;
    return false;
  }
  data->im_size_before_resize_.push_back({im->rows, im->cols});
  data->reshape_order_.push_back("resize");

  cv::resize(
      *im, *im, cv::Size(width_, height_), 0, 0, interpolations[interp_]);
  data->new_im_size_[0] = im->rows;
  data->new_im_size_[1] = im->cols;
  return true;
}

void Transforms::Init(const YAML::Node& transforms_node, bool to_rgb) {
  transforms_.clear();
  to_rgb_ = to_rgb;
  for (const auto& item : transforms_node) {
    std::string name = item.begin()->first.as<std::string>();
    std::cout << "trans name: " << name << std::endl;
    std::shared_ptr<Transform> transform = CreateTransform(name);
    transform->Init(item.begin()->second);
    transforms_.push_back(transform);
  }
}

std::shared_ptr<Transform> Transforms::CreateTransform(
    const std::string& transform_name) {
  if (transform_name == "Normalize") {
    return std::make_shared<Normalize>();
  } else if (transform_name == "ResizeByShort") {
    return std::make_shared<ResizeByShort>();
  } else if (transform_name == "CenterCrop") {
    return std::make_shared<CenterCrop>();
  } else if (transform_name == "Resize") {
    return std::make_shared<Resize>();
  } else if (transform_name == "Padding") {
    return std::make_shared<Padding>();
  } else if (transform_name == "ResizeByLong") {
    return std::make_shared<ResizeByLong>();
  } else {
    std::cerr << "There's unexpected transform(name='" << transform_name
              << "')." << std::endl;
    exit(-1);
  }
}

bool Transforms::Run(cv::Mat* im, ImageBlob* data) {
  // 按照transforms中预处理算子顺序处理图像
  if (to_rgb_) {
    cv::cvtColor(*im, *im, cv::COLOR_BGR2RGB);
  }
  (*im).convertTo(*im, CV_32FC3);
  data->ori_im_size_[0] = im->rows;
  data->ori_im_size_[1] = im->cols;
  data->new_im_size_[0] = im->rows;
  data->new_im_size_[1] = im->cols;
  for (int i = 0; i < transforms_.size(); ++i) {
    if (!transforms_[i]->Run(im, data)) {
      std::cerr << "Apply transforms to image failed!" << std::endl;
      return false;
    }
  }

  // 将图像由NHWC转为NCHW格式
  // 同时转为连续的内存块存储到ImageBlob
  int h = im->rows;
  int w = im->cols;
  int c = im->channels();
  (data->im_data_).resize(c * h * w);
  float* ptr = (data->im_data_).data();
  for (int i = 0; i < c; ++i) {
    cv::extractChannel(*im, cv::Mat(h, w, CV_32FC1, ptr + i * h * w), i);
  }
  return true;
}

}  // namespace PaddleX
