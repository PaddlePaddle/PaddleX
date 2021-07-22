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


#pragma once

#include <vector>

#include "pipeline/include/pipeline.h"
#include "meter_reader/include/meter_config.h"

bool Erode(const int32_t &kernel_size,
           const std::vector<PaddleDeploy::Result> &seg_results,
           std::vector<std::vector<uint8_t>> *seg_label_maps);

bool CircleToRectangle(
  const std::vector<uint8_t> &seg_label_map,
  std::vector<uint8_t> *rectangle_meter);

bool RectangleToLine(const std::vector<uint8_t> &rectangle_meter,
                     std::vector<int> *line_scale,
                     std::vector<int> *line_pointer);

bool MeanBinarization(const std::vector<int> &data,
                      std::vector<int> *binaried_data);

bool LocateScale(const std::vector<int> &scale,
                 std::vector<float> *scale_location);

bool LocatePointer(const std::vector<int> &pointer,
                   float *pointer_location);

bool GetRelativeLocation(
  const std::vector<float> &scale_location,
  const float &pointer_location,
  MeterResult *result);

bool CalculateReading(const MeterResult &result,
                      float *reading);

bool PrintMeterReading(const std::vector<float> &readings);

bool Visualize(const cv::Mat& img,
               const PaddleDeploy::Result &det_result,
               const std::vector<float> &reading,
               cv::Mat* vis_img);

bool GetMeterReading(
  const std::vector<std::vector<uint8_t>> &seg_label_maps,
  std::vector<float> *readings);
