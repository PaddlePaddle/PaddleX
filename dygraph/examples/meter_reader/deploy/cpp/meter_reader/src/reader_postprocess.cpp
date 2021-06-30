// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include <utility>
#include <limits>
#include <cmath>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "meter_reader/include/reader_postprocess.h"

bool Erode(const int32_t &kernel_size,
           const std::vector<PaddleDeploy::Result> &seg_results,
           std::vector<std::vector<uint8_t>> *seg_label_maps) {
  cv::Mat kernel(kernel_size, kernel_size, CV_8U, cv::Scalar(1));
  for (auto result : seg_results) {
    std::vector<uint8_t> label_map(result.seg_result->label_map.data.begin(),
                                   result.seg_result->label_map.data.end());
    cv::Mat mask(result.seg_result->label_map.shape[0],
                 result.seg_result->label_map.shape[1],
                 CV_8UC1,
                 label_map.data());
    cv::erode(mask, mask, kernel);
    std::vector<uint8_t> map;
    if (mask.isContinuous()) {
        map.assign(mask.data, mask.data + mask.total() * mask.channels());
    } else {
      for (int r = 0; r < mask.rows; r++) {
        map.insert(map.end(),
                   mask.ptr<int64_t>(r),
                   mask.ptr<int64_t>(r) + mask.cols * mask.channels());
      }
    }
    seg_label_maps->push_back(map);
  }
  return true;
}


bool CircleToRectangle(
  const std::vector<uint8_t> &seg_label_map,
  std::vector<uint8_t> *rectangle_meter) {
  float theta;
  int rho;
  int image_x;
  int image_y;

  // The minimum scale value is at the bottom left, the maximum scale value
  // is at the bottom right, so the vertical down axis is the starting axis and
  // rotates around the meter ceneter counterclockwise.
  *rectangle_meter =
    std::vector<uint8_t> (RECTANGLE_WIDTH * RECTANGLE_HEIGHT, 0);
  for (int row = 0; row < RECTANGLE_HEIGHT; row++) {
    for (int col = 0; col < RECTANGLE_WIDTH; col++) {
      theta = PI * 2 / RECTANGLE_WIDTH * (col + 1);
      rho = CIRCLE_RADIUS - row - 1;
      int y = static_cast<int>(CIRCLE_CENTER[0] + rho * cos(theta) + 0.5);
      int x = static_cast<int>(CIRCLE_CENTER[1] - rho * sin(theta) + 0.5);
      (*rectangle_meter)[row * RECTANGLE_WIDTH + col] =
        seg_label_map[y * METER_SHAPE[1] + x];
    }
  }

  return true;
}

bool RectangleToLine(const std::vector<uint8_t> &rectangle_meter,
                     std::vector<int> *line_scale,
                     std::vector<int> *line_pointer) {
  // Accumulte the number of positions whose label is 1 along the height axis.
  // Accumulte the number of positions whose label is 2 along the height axis.
  (*line_scale) = std::vector<int> (RECTANGLE_WIDTH, 0);
  (*line_pointer) = std::vector<int> (RECTANGLE_WIDTH, 0);
  for (int col = 0; col < RECTANGLE_WIDTH; col++) {
    for (int row = 0; row < RECTANGLE_HEIGHT; row++) {
        if (rectangle_meter[row * RECTANGLE_WIDTH + col] ==
          static_cast<uint8_t>(SEG_CNAME2CLSID["pointer"])) {
            (*line_pointer)[col]++;
        } else if (rectangle_meter[row * RECTANGLE_WIDTH + col] ==
          static_cast<uint8_t>(SEG_CNAME2CLSID["scale"])) {
            (*line_scale)[col]++;
        }
    }
  }
  return true;
}

bool MeanBinarization(const std::vector<int> &data,
                      std::vector<int> *binaried_data) {
  int sum = 0;
  float mean = 0;
  for (auto i = 0; i < data.size(); i++) {
    sum = sum + data[i];
  }
  mean = static_cast<float>(sum) / static_cast<float>(data.size());

  for (auto i = 0; i < data.size(); i++) {
    if (static_cast<float>(data[i]) >= mean) {
      binaried_data->push_back(1);
    } else {
      binaried_data->push_back(0);
    }
  }
  return  true;
}

bool LocateScale(const std::vector<int> &scale,
                 std::vector<float> *scale_location) {
  float one_scale_location = 0;
  bool find_start = false;
  int one_scale_start = 0;
  int one_scale_end = 0;

  for (int i = 0; i < RECTANGLE_WIDTH; i++) {
    // scale location
    if (scale[i] > 0 && scale[i + 1] > 0) {
      if (!find_start) {
        one_scale_start = i;
        find_start = true;
      }
    }
    if (find_start) {
      if (scale[i] == 0 && scale[i + 1] == 0) {
          one_scale_end = i - 1;
          one_scale_location = (one_scale_start + one_scale_end) / 2.;
          scale_location->push_back(one_scale_location);
          one_scale_start = 0;
          one_scale_end = 0;
          find_start = false;
      }
    }
  }
  return true;
}

bool LocatePointer(const std::vector<int> &pointer,
                   float *pointer_location) {
  bool find_start = false;
  int one_pointer_start = 0;
  int one_pointer_end = 0;

  for (int i = 0; i < RECTANGLE_WIDTH; i++) {
    // pointer location
    if (pointer[i] > 0 && pointer[i + 1] > 0) {
      if (!find_start) {
        one_pointer_start = i;
        find_start = true;
      }
    }
    if (find_start) {
      if ((pointer[i] == 0) && (pointer[i+1] == 0)) {
        one_pointer_end = i - 1;
        *pointer_location = (one_pointer_start + one_pointer_end) / 2.;
        one_pointer_start = 0;
        one_pointer_end = 0;
        find_start = false;
        break;
      }
    }
  }
  return true;
}

bool GetRelativeLocation(
  const std::vector<float> &scale_location,
  const float &pointer_location,
  MeterResult *result) {
  int num_scales = static_cast<int>(scale_location.size());
  result->num_scales_ = num_scales;
  result->pointed_scale_ = -1;
  if (num_scales > 0) {
    for (auto i = 0; i < num_scales - 1; i++) {
      if (scale_location[i] <= pointer_location &&
            pointer_location < scale_location[i + 1]) {
        result->pointed_scale_ = i + 1 +
          (pointer_location-scale_location[i]) /
          (scale_location[i+1]-scale_location[i] + 1e-05);
      }
    }
  }
  return true;
}

bool CalculateReading(const MeterResult &result,
                      float *reading) {
  // Provide a digital readout according to point location relative
  // to the scales
  if (result.num_scales_ > TYPE_THRESHOLD) {
    *reading = result.pointed_scale_ * METER_CONFIG[0].scale_interval_value_;
  } else {
    *reading = result.pointed_scale_ * METER_CONFIG[1].scale_interval_value_;
  }
  return true;
}

bool PrintMeterReading(const std::vector<float> &readings) {
  for (auto i = 0; i < readings.size(); ++i) {
    std::cout << "Meter " << i + 1 << ": " << readings[i] << std::endl;
  }
  return true;
}

bool Visualize(const cv::Mat& img,
               const PaddleDeploy::Result &det_result,
               const std::vector<float> &reading,
               cv::Mat* vis_img) {
  for (auto i = 0; i < det_result.det_result->boxes.size(); ++i) {
     std::string category = std::to_string(reading[i]);
     det_result.det_result->boxes[i].category = category;
  }

  PaddleDeploy::Visualize(img, *(det_result.det_result), vis_img);
  return true;
}

bool GetMeterReading(
  const std::vector<std::vector<uint8_t>> &seg_label_maps,
  std::vector<float> *readings) {
  for (auto i = 0; i < seg_label_maps.size(); i++) {
    std::vector<uint8_t> rectangle_meter;
    CircleToRectangle(seg_label_maps[i], &rectangle_meter);

    std::vector<int> line_scale;
    std::vector<int> line_pointer;
    RectangleToLine(rectangle_meter, &line_scale, &line_pointer);

    std::vector<int> binaried_scale;
    MeanBinarization(line_scale, &binaried_scale);
    std::vector<int> binaried_pointer;
    MeanBinarization(line_pointer, &binaried_pointer);

    std::vector<float> scale_location;
    LocateScale(binaried_scale, &scale_location);

    float pointer_location;
    LocatePointer(binaried_pointer, &pointer_location);

    MeterResult result;
    GetRelativeLocation(
      scale_location, pointer_location, &result);

    float reading;
    CalculateReading(result, &reading);
    readings->push_back(reading);
  }
  return true;
}
