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

#include "include/paddlex/transforms.h"

#include <math.h>

#include <iostream>
#include <string>
#include <vector>

namespace PaddleX {

std::map<std::string, int> interpolations = {{"LINEAR", cv::INTER_LINEAR},
                                             {"NEAREST", cv::INTER_NEAREST},
                                             {"AREA", cv::INTER_AREA},
                                             {"CUBIC", cv::INTER_CUBIC},
                                             {"LANCZOS4", cv::INTER_LANCZOS4}};

bool Normalize::Run(cv::Mat* im, ImageBlob* data) {
  std::vector<float> range_val;
  for (int c = 0; c < im->channels(); c++) {
    range_val.push_back(max_val_[c] - min_val_[c]);
  }

  std::vector<cv::Mat> split_im;
  cv::split(*im, split_im);
  #pragma omp parallel for num_threads(im->channels())
  for (int c = 0; c < im->channels(); c++) {
    float range_val = max_val_[c] - min_val_[c];
    cv::subtract(split_im[c], cv::Scalar(min_val_[c]), split_im[c]);
    cv::divide(split_im[c], cv::Scalar(range_val), split_im[c]);
    cv::subtract(split_im[c], cv::Scalar(mean_[c]), split_im[c]);
    cv::divide(split_im[c], cv::Scalar(std_[c]), split_im[c]);
  }
  cv::merge(split_im, *im);
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
  int width = static_cast<int>(round(scale * im->cols));
  int height = static_cast<int>(round(scale * im->rows));
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

void Padding::GeneralPadding(cv::Mat* im,
                             const std::vector<float> &padding_val,
                             int padding_w, int padding_h) {
  cv::Scalar value;
  if (im->channels() == 1) {
    value = cv::Scalar(padding_val[0]);
  } else if (im->channels() == 2) {
    value = cv::Scalar(padding_val[0], padding_val[1]);
  } else if (im->channels() == 3) {
    value = cv::Scalar(padding_val[0], padding_val[1], padding_val[2]);
  } else if (im->channels() == 4) {
    value = cv::Scalar(padding_val[0], padding_val[1], padding_val[2],
                                  padding_val[3]);
  }
  cv::copyMakeBorder(
  *im,
  *im,
  0,
  padding_h,
  0,
  padding_w,
  cv::BORDER_CONSTANT,
  value);
}

void Padding::MultichannelPadding(cv::Mat* im,
                                  const std::vector<float> &padding_val,
                                  int padding_w, int padding_h) {
  std::vector<cv::Mat> padded_im_per_channel(im->channels());
  #pragma omp parallel for num_threads(im->channels())
  for (size_t i = 0; i < im->channels(); i++) {
    const cv::Mat per_channel = cv::Mat(im->rows + padding_h,
                                        im->cols + padding_w,
                                        CV_32FC1,
                                        cv::Scalar(padding_val[i]));
    padded_im_per_channel[i] = per_channel;
  }
  cv::Mat padded_im;
  cv::merge(padded_im_per_channel, padded_im);
  cv::Rect im_roi = cv::Rect(0, 0, im->cols, im->rows);
  im->copyTo(padded_im(im_roi));
  *im = padded_im;
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
  if (im->channels() < 5) {
    Padding::GeneralPadding(im, im_value_, padding_w, padding_h);
  } else {
    Padding::MultichannelPadding(im, im_value_, padding_w, padding_h);
  }
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

bool Clip::Run(cv::Mat* im, ImageBlob* data) {
  std::vector<cv::Mat> split_im;
  cv::split(*im, split_im);
  for (int c = 0; c < im->channels(); c++) {
    cv::threshold(split_im[c], split_im[c], max_val_[c], max_val_[c],
                  cv::THRESH_TRUNC);
    cv::subtract(cv::Scalar(0), split_im[c], split_im[c]);
    cv::threshold(split_im[c], split_im[c], min_val_[c], min_val_[c],
                  cv::THRESH_TRUNC);
    cv::divide(split_im[c], cv::Scalar(-1), split_im[c]);
  }
  cv::merge(split_im, *im);
  return true;
}

void Transforms::Init(const YAML::Node& transforms_node, bool to_rgb) {
  transforms_.clear();
  to_rgb_ = to_rgb;
  for (const auto& item : transforms_node) {
    std::string name = item.begin()->first.as<std::string>();
    if (name == "ArrangeClassifier") {
      continue;
    }
    if (name == "ArrangeSegmenter") {
      continue;
    }
    if (name == "ArrangeFasterRCNN") {
      continue;
    }
    if (name == "ArrangeMaskRCNN") {
      continue;
    }
    if (name == "ArrangeYOLOv3") {
      continue;
    }

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
  } else if (transform_name == "Clip") {
    return std::make_shared<Clip>();
  } else {
    std::cerr << "There's unexpected transform(name='" << transform_name
              << "')." << std::endl;
    exit(-1);
  }
}

bool Transforms::Run(cv::Mat* im, ImageBlob* data) {
  // do all preprocess ops by order
  if (to_rgb_) {
    cv::cvtColor(*im, *im, cv::COLOR_BGR2RGB);
  }
  (*im).convertTo(*im, CV_32FC(im->channels()));
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

  // data format NHWC to NCHW
  // img data save to ImageBlob
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
