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
#include <sys/io.h>
#include <sys/stat.h>
#include <sys/types.h>
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

// Generate visualization colormap for each class
std::vector<int> GenerateColorMap(int num_class);

cv::Mat Visualize(const cv::Mat& img,
                     const DetResult& results,
                     const std::map<int, std::string>& labels,
                     const std::vector<int>& colormap,
                     float threshold = 0.5);

cv::Mat Visualize(const cv::Mat& img,
                     const SegResult& result,
                     const std::map<int, std::string>& labels,
                     const std::vector<int>& colormap);

std::string generate_save_path(const std::string& save_dir,
                               const std::string& file_path);
}  // namespce of PaddleX
