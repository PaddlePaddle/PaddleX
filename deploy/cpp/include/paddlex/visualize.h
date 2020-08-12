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

#include <iostream>
#include <map>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else  // Linux/Unix
#include <dirent.h>
// #include <sys/io.h>
#if defined(__arm__) || defined(__aarch64__)  // for arm
#include <aarch64-linux-gnu/sys/stat.h>
#include <aarch64-linux-gnu/sys/types.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif
#include <unistd.h>
#endif
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/paddlex/results.h"

#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

namespace PaddleX {

/*
 * @brief
 * Generate visualization colormap for each class
 *
 * @param number of class
 * @return color map, the size of vector is 3 * num_class
 * */
std::vector<int> GenerateColorMap(int num_class);


/*
 * @brief
 * Visualize the detection result
 *
 * @param img: initial image matrix
 * @param results: the detection result
 * @param labels: label map
 * @param threshold: minimum confidence to display
 * @return visualized image matrix
 * */
cv::Mat Visualize(const cv::Mat& img,
                     const DetResult& results,
                     const std::map<int, std::string>& labels,
                     float threshold = 0.5);

/*
 * @brief
 * Visualize the segmentation result
 *
 * @param img: initial image matrix
 * @param results: the detection result
 * @param labels: label map
 * @return visualized image matrix
 * */
cv::Mat Visualize(const cv::Mat& img,
                     const SegResult& result,
                     const std::map<int, std::string>& labels);

/*
 * @brief
 * generate save path for visualized image matrix
 *
 * @param save_dir: directory for saving visualized image matrix
 * @param file_path: sourcen image file path
 * @return path of saving visualized result
 * */
std::string generate_save_path(const std::string& save_dir,
                               const std::string& file_path);
}  // namespace PaddleX
