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

#include <glog/logging.h>
#include <omp.h>

#include <algorithm>
#include <chrono>  // NOLINT
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <ctime>
#include "include/paddlex/paddlex.h"
#include "include/paddlex/visualize.h"

#if defined(__arm__) || defined(__aarch64__)
#include <opencv2/videoio/legacy/constants_c.h>
#endif

using namespace std::chrono;  // NOLINT

DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_bool(use_trt, false, "Infering with TensorRT");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_string(key, "", "key of encryption");
DEFINE_string(image, "", "Path of test image file");
DEFINE_bool(use_camera, false, "Infering with Camera");
DEFINE_int32(camera_id, 0, "Camera id");
DEFINE_string(video_path, "", "Path of input video");
DEFINE_bool(show_result, false, "show the result of each frame with a window");
DEFINE_bool(save_result, true, "save the result of each frame to a video");
DEFINE_string(save_dir, "output", "Path to save visualized image");

int main(int argc, char** argv) {
  // 解析命令行参数
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_dir == "") {
    std::cerr << "--model_dir need to be defined" << std::endl;
    return -1;
  }
  if (FLAGS_image == "" & FLAGS_video_path == ""
      & FLAGS_use_camera == false) {
    std::cerr << "--image or --video_path or --use_camera need to be defined"
              << std::endl;
    return -1;
  }

  // 加载模型
  PaddleX::Model model;
  model.Init(FLAGS_model_dir,
             FLAGS_use_gpu,
             FLAGS_use_trt,
             FLAGS_gpu_id,
             FLAGS_key);
  if (FLAGS_use_camera || FLAGS_video_path != "") {
    // 打开视频流
    cv::VideoCapture capture;
    if (FLAGS_use_camera) {
      capture.open(FLAGS_camera_id);
      if (!capture.isOpened()) {
        std::cout << "Can not open the camera "
                  << FLAGS_camera_id << "."
                  << std::endl;
        return -1;
      }
    } else {
      capture.open(FLAGS_video_path);
      if (!capture.isOpened()) {
        std::cout << "Can not open the video "
                  << FLAGS_video_path << "."
                  << std::endl;
        return -1;
      }
    }

    // 创建VideoWriter
    cv::VideoWriter video_out;
    std::string video_out_path;
    if (FLAGS_save_result) {
      // 获取视频流信息: 分辨率, 帧率
      int video_width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
      int video_height =
        static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
      int video_fps = static_cast<int>(capture.get(CV_CAP_PROP_FPS));
      int video_fourcc;
      if (FLAGS_use_camera) {
        video_fourcc = 828601953;
      } else {
        video_fourcc = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));
      }
      if (FLAGS_use_camera) {
        time_t now = time(0);
        video_out_path =
          PaddleX::generate_save_path(FLAGS_save_dir,
                                      std::to_string(now) + ".mp4");
      } else {
        video_out_path =
          PaddleX::generate_save_path(FLAGS_save_dir, FLAGS_video_path);
      }
      video_out.open(video_out_path.c_str(),
                     video_fourcc,
                     video_fps,
                     cv::Size(video_width, video_height),
                     true);
      if (!video_out.isOpened()) {
        std::cout << "Create video writer failed!" << std::endl;
        return -1;
      }
    }

    PaddleX::SegResult result;
    cv::Mat frame;
    int key;
    while (capture.read(frame)) {
      if (FLAGS_show_result || FLAGS_use_camera) {
       key = cv::waitKey(1);
       // 按下ESC退出整个程序，保存视频文件到磁盘
       if (key == 27) {
         break;
       }
      } else if (frame.empty()) {
        break;
      }
      // 开始预测
      model.predict(frame, &result);
      // 可视化
      std::vector<uint8_t> label_map(result.label_map.data.begin(),
                                     result.label_map.data.end());
      cv::Mat mask(result.label_map.shape[0],
                   result.label_map.shape[1],
                   CV_8UC1,
                   label_map.data());
      int rows = result.label_map.shape[0];
      int cols = result.label_map.shape[1];
      cv::Mat vis_img = frame.clone();
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          int category_id = static_cast<int>(mask.at<uchar>(i, j));
          if (category_id == 0) {
            vis_img.at<cv::Vec3b>(i, j)[0] = 255;
            vis_img.at<cv::Vec3b>(i, j)[1] = 255;
            vis_img.at<cv::Vec3b>(i, j)[2] = 255;
          }
        }
      }
      if (FLAGS_show_result || FLAGS_use_camera) {
        cv::imshow("human_seg", vis_img);
      }
      if (FLAGS_save_result) {
        video_out.write(vis_img);
      }
      result.clear();
    }
    capture.release();
    if (FLAGS_save_result) {
      video_out.release();
      std::cout << "Visualized output saved as " << video_out_path << std::endl;
    }
    if (FLAGS_show_result || FLAGS_use_camera) {
      cv::destroyAllWindows();
    }
  } else {
    PaddleX::SegResult result;
    cv::Mat im = cv::imread(FLAGS_image, 1);
    model.predict(im, &result);
    // 可视化
    std::vector<uint8_t> label_map(result.label_map.data.begin(),
                                   result.label_map.data.end());
    cv::Mat mask(result.label_map.shape[0],
                 result.label_map.shape[1],
                 CV_8UC1,
                 label_map.data());
    int rows = result.label_map.shape[0];
    int cols = result.label_map.shape[1];
    cv::Mat vis_img = im.clone();
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        int category_id = static_cast<int>(mask.at<uchar>(i, j));
        if (category_id == 0) {
          vis_img.at<cv::Vec3b>(i, j)[0] = 255;
          vis_img.at<cv::Vec3b>(i, j)[1] = 255;
          vis_img.at<cv::Vec3b>(i, j)[2] = 255;
        }
      }
    }
    std::string save_path =
        PaddleX::generate_save_path(FLAGS_save_dir, FLAGS_image);
    cv::imwrite(save_path, vis_img);
    result.clear();
    std::cout << "Visualized output saved as " << save_path << std::endl;
  }
  return 0;
}
