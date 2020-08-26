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

#include <glog/logging.h>
#include <omp.h>

#include <algorithm>
#include <chrono>  // NOLINT
#include <iostream>
#include <vector>
#include <utility>
#include <limits>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "meter_reader/global.h"
#include "meter_reader/postprocess.h"
#include "include/paddlex/paddlex.h"
#include "include/paddlex/visualize.h"

using namespace std::chrono;  // NOLINT

DEFINE_string(det_model_dir, "", "Path of detection inference model");
DEFINE_string(seg_model_dir, "", "Path of segmentation inference model");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_bool(use_trt, false, "Infering with TensorRT");
DEFINE_bool(use_camera, false, "Infering with Camera");
DEFINE_bool(use_erode, true, "Eroding predicted label map");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_int32(camera_id, 0, "Camera id");
DEFINE_int32(thread_num,
             omp_get_num_procs(),
             "Number of preprocessing threads");
DEFINE_int32(erode_kernel, true, "Eroding kernel size");
DEFINE_int32(seg_batch_size, 2, "Batch size of segmentation infering");
DEFINE_string(det_key, "", "Detector key of encryption");
DEFINE_string(seg_key, "", "Segmenter model key of encryption");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image list file");
DEFINE_string(save_dir, "output", "Path to save visualized image");
DEFINE_double(score_threshold, 0.5,
  "Detected bbox whose score is lower than this threshlod is filtered");

void predict(const cv::Mat &input_image, PaddleX::Model *det_model,
             PaddleX::Model *seg_model, const std::string save_dir,
             const std::string image_path, const bool use_erode,
             const int erode_kernel, const int thread_num,
             const int seg_batch_size, const double threshold) {
  PaddleX::DetResult det_result;
  det_model->predict(input_image, &det_result);

  PaddleX::DetResult filter_result;
  int num_bboxes = det_result.boxes.size();
  for (int i = 0; i < num_bboxes; ++i) {
    double score = det_result.boxes[i].score;
    if (score > threshold || score == threshold) {
      PaddleX::Box box;
      box.category_id = det_result.boxes[i].category_id;
      box.category = det_result.boxes[i].category;
      box.score = det_result.boxes[i].score;
      box.coordinate = det_result.boxes[i].coordinate;
      filter_result.boxes.push_back(std::move(box));
    }
  }

  int meter_num = filter_result.boxes.size();
  if (!meter_num) {
      std::cout << "Don't find any meter." << std::endl;
      return;
  }

  std::vector<std::vector<int64_t>> seg_result(meter_num);
  for (int i = 0; i < meter_num; i += seg_batch_size) {
    int im_vec_size =
      std::min(static_cast<int>(meter_num), i + seg_batch_size);
    std::vector<cv::Mat> meters_image(im_vec_size - i);
    int batch_thread_num = std::min(thread_num, im_vec_size - i);
    #pragma omp parallel for num_threads(batch_thread_num)
    for (int j = i; j < im_vec_size; ++j) {
      int left = static_cast<int>(filter_result.boxes[j].coordinate[0]);
      int top = static_cast<int>(filter_result.boxes[j].coordinate[1]);
      int width = static_cast<int>(filter_result.boxes[j].coordinate[2]);
      int height = static_cast<int>(filter_result.boxes[j].coordinate[3]);
      int right = left + width - 1;
      int bottom = top + height - 1;

      cv::Mat sub_image = input_image(
        cv::Range(top, bottom + 1), cv::Range(left, right + 1));
      float scale_x =
        static_cast<float>(METER_SHAPE[0]) / static_cast<float>(sub_image.cols);
      float scale_y =
        static_cast<float>(METER_SHAPE[1]) / static_cast<float>(sub_image.rows);
      cv::resize(sub_image,
                 sub_image,
                 cv::Size(),
                 scale_x,
                 scale_y,
                 cv::INTER_LINEAR);
      meters_image[j - i] = std::move(sub_image);
    }
    std::vector<PaddleX::SegResult> batch_result(im_vec_size - i);
    seg_model->predict(meters_image, &batch_result, batch_thread_num);
    #pragma omp parallel for num_threads(batch_thread_num)
    for (int j = i; j < im_vec_size; ++j) {
      if (use_erode) {
        cv::Mat kernel(4, 4, CV_8U, cv::Scalar(1));
        std::vector<uint8_t> label_map(
          batch_result[j - i].label_map.data.begin(),
          batch_result[j - i].label_map.data.end());
        cv::Mat mask(batch_result[j - i].label_map.shape[0],
                     batch_result[j - i].label_map.shape[1],
                     CV_8UC1,
                     label_map.data());
        cv::erode(mask, mask, kernel);
        std::vector<int64_t> map;
        if (mask.isContinuous()) {
            map.assign(mask.data, mask.data + mask.total() * mask.channels());
        } else {
          for (int r = 0; r < mask.rows; r++) {
            map.insert(map.end(),
                       mask.ptr<int64_t>(r),
                       mask.ptr<int64_t>(r) + mask.cols * mask.channels());
          }
        }
        seg_result[j] = std::move(map);
      } else {
        seg_result[j] = std::move(batch_result[j - i].label_map.data);
      }
    }
  }

  std::vector<READ_RESULT> read_results(meter_num);
  int all_thread_num = std::min(thread_num, meter_num);
  read_process(seg_result, &read_results, all_thread_num);

  cv::Mat output_image = input_image.clone();
  for (int i = 0; i < meter_num; i++) {
    float result = 0;;
    if (read_results[i].scale_num > TYPE_THRESHOLD) {
      result = read_results[i].scales * meter_config[0].scale_value;
    } else {
      result = read_results[i].scales * meter_config[1].scale_value;
    }
    std::cout << "-- Meter " << i
              << " -- result: " << result
              << " --" << std::endl;

    int lx = static_cast<int>(filter_result.boxes[i].coordinate[0]);
    int ly = static_cast<int>(filter_result.boxes[i].coordinate[1]);
    int w = static_cast<int>(filter_result.boxes[i].coordinate[2]);
    int h = static_cast<int>(filter_result.boxes[i].coordinate[3]);

    cv::Rect bounding_box = cv::Rect(lx, ly, w, h) &
        cv::Rect(0, 0, output_image.cols, output_image.rows);
    if (w > 0 && h > 0) {
      cv::Scalar color = cv::Scalar(237, 189, 101);
      cv::rectangle(output_image, bounding_box, color);
      cv::rectangle(output_image,
                    cv::Point2d(lx, ly),
                    cv::Point2d(lx + w, ly - 30),
                    color, -1);

      std::string class_name = "Meter";
      cv::putText(output_image,
                  class_name + " " + std::to_string(result),
                  cv::Point2d(lx, ly-5),
                  cv::FONT_HERSHEY_SIMPLEX,
                  1, cv::Scalar(255, 255, 255), 2);
    }
  }

  cv::Mat result_image;
  cv::Size resize_size(RESULT_SHAPE[0], RESULT_SHAPE[1]);
  cv::resize(output_image, result_image, resize_size, 0, 0, cv::INTER_LINEAR);
  std::string save_path = PaddleX::generate_save_path(save_dir, image_path);
  cv::imwrite(save_path, result_image);

  return;
}


int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_det_model_dir == "") {
    std::cerr << "--det_model_dir need to be defined" << std::endl;
    return -1;
  }
  if (FLAGS_seg_model_dir == "") {
    std::cerr << "--seg_model_dir need to be defined" << std::endl;
    return -1;
  }
  if (FLAGS_image == "" & FLAGS_image_list == "" & FLAGS_use_camera == false) {
    std::cerr << "--image or --image_list need to be defined "
              << "when the camera is not been used" << std::endl;
    return -1;
  }

  // Load model
  PaddleX::Model det_model;
  det_model.Init(FLAGS_det_model_dir, FLAGS_use_gpu, FLAGS_use_trt,
                 FLAGS_gpu_id, FLAGS_det_key);
  PaddleX::Model seg_model;
  seg_model.Init(FLAGS_seg_model_dir, FLAGS_use_gpu, FLAGS_use_trt,
                 FLAGS_gpu_id, FLAGS_seg_key);

  double total_running_time_s = 0.0;
  double total_imread_time_s = 0.0;
  int imgs = 1;
  if (FLAGS_use_camera) {
    cv::VideoCapture cap(FLAGS_camera_id);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_SHAPE[0]);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_SHAPE[1]);
    if (!cap.isOpened()) {
      std::cout << "Open the camera unsuccessfully." << std::endl;
      return -1;
    }
    std::cout << "Open the camera successfully." << std::endl;

    while (1) {
      auto start = system_clock::now();
      cv::Mat im;
      cap >> im;
      auto imread_end = system_clock::now();
      std::cout << "-------------------------" << std::endl;
      std::cout << "Got a camera image." << std::endl;
      std::string ext_name = ".jpg";
      predict(im, &det_model, &seg_model, FLAGS_save_dir,
              std::to_string(imgs) + ext_name, FLAGS_use_erode,
              FLAGS_erode_kernel, FLAGS_thread_num,
              FLAGS_seg_batch_size, FLAGS_score_threshold);
      imgs++;
      auto imread_duration = duration_cast<microseconds>(imread_end - start);
      total_imread_time_s += static_cast<double>(imread_duration.count()) *
                             microseconds::period::num /
                             microseconds::period::den;

      auto end = system_clock::now();
      auto duration = duration_cast<microseconds>(end - start);
      total_running_time_s += static_cast<double>(duration.count()) *
                              microseconds::period::num /
                              microseconds::period::den;
    }
    cap.release();
    cv::destroyAllWindows();
  } else {
    if (FLAGS_image_list != "") {
      std::ifstream inf(FLAGS_image_list);
      if (!inf) {
        std::cerr << "Fail to open file " << FLAGS_image_list << std::endl;
        return -1;
      }
      std::string image_path;
      while (getline(inf, image_path)) {
        auto start = system_clock::now();
        cv::Mat im = cv::imread(image_path, 1);
        imgs++;
        auto imread_end = system_clock::now();

        predict(im, &det_model, &seg_model, FLAGS_save_dir,
                image_path, FLAGS_use_erode, FLAGS_erode_kernel,
                FLAGS_thread_num, FLAGS_seg_batch_size,
                FLAGS_score_threshold);

        auto imread_duration = duration_cast<microseconds>(imread_end - start);
        total_imread_time_s += static_cast<double>(imread_duration.count()) *
                               microseconds::period::num /
                               microseconds::period::den;

        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        total_running_time_s += static_cast<double>(duration.count()) *
                                microseconds::period::num /
                                microseconds::period::den;
      }
    } else {
      auto start = system_clock::now();
      cv::Mat im = cv::imread(FLAGS_image, 1);
      auto imread_end = system_clock::now();

      predict(im, &det_model, &seg_model, FLAGS_save_dir,
              FLAGS_image, FLAGS_use_erode, FLAGS_erode_kernel,
              FLAGS_thread_num, FLAGS_seg_batch_size,
              FLAGS_score_threshold);

      auto imread_duration = duration_cast<microseconds>(imread_end - start);
      total_imread_time_s += static_cast<double>(imread_duration.count()) *
                             microseconds::period::num /
                             microseconds::period::den;

      auto end = system_clock::now();
      auto duration = duration_cast<microseconds>(end - start);
      total_running_time_s += static_cast<double>(duration.count()) *
                              microseconds::period::num /
                              microseconds::period::den;
    }
  }
  std::cout << "Total running time: " << total_running_time_s
            << " s, average running time: " << total_running_time_s / imgs
            << " s/img, total read img time: " << total_imread_time_s
            << " s, average read time: " << total_imread_time_s / imgs
            << " s/img" << std::endl;
  return 0;
}
