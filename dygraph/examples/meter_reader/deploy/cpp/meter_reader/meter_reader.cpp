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

#include <gflags/gflags.h>
#include <string>
#include <iostream>
#include <vector>
#include <utility>
#include <limits>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "pipeline/include/pipeline.h"
#include "meter_reader/include/reader_postprocess.h"

DEFINE_string(pipeline_cfg, "", "Path of pipeline config file");
DEFINE_bool(use_erode, true, "Eroding predicted label map");
DEFINE_int32(erode_kernel, 4, "Eroding kernel size");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(save_dir, "", "Path to save visualized results");

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_pipeline_cfg == "") {
    std::cerr << "--pipeline_cfg need to be defined" << std::endl;
    return -1;
  }
  if (FLAGS_image == "") {
    std::cerr << "--image need to be defined "
              << "when the camera is not been used" << std::endl;
    return -1;
  }

  std::vector<std::string> image_paths = {FLAGS_image};
  PaddleXPipeline::Pipeline pipeline;
  if (pipeline.Init(FLAGS_pipeline_cfg)) {
    pipeline.SetInput("src0", image_paths);
    pipeline.Run();
    std::vector<PaddleDeploy::Result> det_results;
    std::vector<PaddleDeploy::Result> seg_results;
    pipeline.GetOutput("sink0", &det_results);
    pipeline.GetOutput("sink1", &seg_results);

    // Do image erosion for the predicted label map of each meter
    std::vector<std::vector<uint8_t>> seg_label_maps;
    Erode(FLAGS_erode_kernel, seg_results, &seg_label_maps);

    // The postprocess are done to get the reading or each meter
    std::vector<float> readings;
    GetMeterReading(seg_label_maps, &readings);
    PrintMeterReading(readings);
    if (FLAGS_save_dir != "") {
      cv::Mat img = cv::imread(FLAGS_image);
      cv::Mat vis_img;
      Visualize(img, det_results[0], readings, &vis_img);
      std::string save_path;
      if (PaddleXPipeline::GenerateSavePath(
          FLAGS_save_dir, FLAGS_image, &save_path)) {
         cv::imwrite(save_path, vis_img);
      }
    }
  }

  return 0;
}
