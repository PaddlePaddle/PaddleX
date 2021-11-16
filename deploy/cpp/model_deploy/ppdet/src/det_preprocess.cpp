// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "model_deploy/ppdet/include/det_preprocess.h"

namespace PaddleDeploy {

bool DetPreprocess::Init(const YAML::Node& yaml_config) {
  if (!BuildTransform(yaml_config)) return false;
  if (!yaml_config["model_name"].IsDefined()) {
    std::cerr << "Yaml file no model_name" << std::endl;
    return false;
  }
  version_ = yaml_config["version"].as<std::string>();
  model_arch_ = yaml_config["model_name"].as<std::string>();
  return true;
}

bool DetPreprocess::PrepareInputs(const std::vector<ShapeInfo>& shape_infos,
                                  std::vector<cv::Mat>* imgs,
                                  std::vector<DataBlob>* inputs,
                                  int thread_num) {
  inputs->clear();
  if (!PreprocessImages(shape_infos, imgs, thread_num = thread_num)) {
    std::cerr << "Error happend while execute function "
              << "DetPreprocess::Run" << std::endl;
    return false;
  }

  if (version_ >= "2.0") {
    return PrepareInputsForV2(*imgs, shape_infos, inputs, thread_num);
  }

  if (model_arch_.find("YOLO") != std::string::npos) {
    return PrepareInputsForYOLO(*imgs, shape_infos, inputs, thread_num);
  }
  if (model_arch_.find("RCNN") != std::string::npos) {
    return PrepareInputsForRCNN(*imgs, shape_infos, inputs, thread_num);
  }
  std::cerr << "Unsupported model type of '" << model_arch_ << "' "
            << std::endl;
  return false;
}

bool DetPreprocess::PrepareInputsForV2(
    const std::vector<cv::Mat>& imgs, const std::vector<ShapeInfo>& shape_infos,
    std::vector<DataBlob>* inputs, int thread_num) {
  DataBlob scale_factor("scale_factor");
  DataBlob image("image");
  DataBlob im_shape("im_shape");
  // TODO(jiangjiajun): only 3 channel supported
  int batch = imgs.size();
  int w = shape_infos[0].shapes.back()[0];
  int h = shape_infos[0].shapes.back()[1];

  scale_factor.Resize({batch, 2}, FLOAT32);
  image.Resize({batch, 3, h, w}, FLOAT32);
  im_shape.Resize({batch, 2}, FLOAT32);

  int sample_shape = 3 * h * w;
  #pragma omp parallel for num_threads(thread_num)
  for (auto i = 0; i < batch; ++i) {
    int shapes_num = shape_infos[i].shapes.size();
    float origin_w = static_cast<float>(shape_infos[i].shapes[0][0]);
    float origin_h = static_cast<float>(shape_infos[i].shapes[0][1]);
    float resize_w = origin_w;
    float resize_h = origin_h;
    for (auto j = shapes_num - 1; j > 1; --j) {
      if (shape_infos[i].transforms[j] == "Padding") {
        continue;
      }
      resize_w = static_cast<float>(shape_infos[i].shapes[j][0]);
      resize_h = static_cast<float>(shape_infos[i].shapes[j][1]);
      if (shape_infos[i].transforms[j].rfind("Resize", 0) == 0)
        break;
    }
    float scale_x = resize_w / origin_w;
    float scale_y = resize_h / origin_h;
    float scale_factor_data[] = {scale_y, scale_x};
    float im_shape_data[] = {resize_h, resize_w};
    memcpy(image.data.data() + i * sample_shape * sizeof(float), imgs[i].data,
           sample_shape * sizeof(float));
    memcpy(im_shape.data.data() + i * 2 * sizeof(float), im_shape_data,
           2 * sizeof(float));
    memcpy(scale_factor.data.data() + i * 2 * sizeof(float), scale_factor_data,
           2 * sizeof(float));
  }

  inputs->clear();
  inputs->push_back(std::move(im_shape));
  inputs->push_back(std::move(image));
  inputs->push_back(std::move(scale_factor));
  return true;
}

bool DetPreprocess::PrepareInputsForYOLO(
    const std::vector<cv::Mat>& imgs, const std::vector<ShapeInfo>& shape_infos,
    std::vector<DataBlob>* inputs, int thread_num) {
  DataBlob im("image");
  DataBlob im_size("im_size");
  // TODO(jiangjiajun): only 3 channel supported
  int batch = imgs.size();
  int w = shape_infos[0].shapes.back()[0];
  int h = shape_infos[0].shapes.back()[1];

  im.Resize({batch, 3, h, w}, FLOAT32);
  im_size.Resize({batch, 2}, INT32);

  int sample_shape = 3 * h * w;
  #pragma omp parallel for num_threads(thread_num)
  for (auto i = 0; i < batch; ++i) {
    memcpy(im.data.data() + i * sample_shape * sizeof(float), imgs[i].data,
           sample_shape * sizeof(float));
    int data[2] = {shape_infos[i].shapes[0][1], shape_infos[i].shapes[0][0]};
    memcpy(im_size.data.data() + i * 2 * sizeof(int), data, 2 * sizeof(int));
  }

  inputs->clear();
  inputs->push_back(std::move(im));
  inputs->push_back(std::move(im_size));
  return true;
}

bool DetPreprocess::PrepareInputsForRCNN(
    const std::vector<cv::Mat>& imgs, const std::vector<ShapeInfo>& shape_infos,
    std::vector<DataBlob>* inputs, int thread_num) {
  DataBlob im("image");
  DataBlob im_info("im_info");
  DataBlob im_shape("im_shape");
  // TODO(jiangjiajun): only 3 channel supported
  int batch = imgs.size();
  int w = shape_infos[0].shapes.back()[0];
  int h = shape_infos[0].shapes.back()[1];

  im.Resize({batch, 3, h, w}, FLOAT32);
  im_info.Resize({batch, 3}, FLOAT32);
  im_shape.Resize({batch, 3}, FLOAT32);

  int sample_shape = 3 * h * w;
  #pragma omp parallel for num_threads(thread_num)
  for (auto i = 0; i < batch; ++i) {
    int shapes_num = shape_infos[i].shapes.size();
    float origin_w = static_cast<float>(shape_infos[i].shapes[0][0]);
    float origin_h = static_cast<float>(shape_infos[i].shapes[0][1]);
    float resize_w = origin_w;
    for (auto j = shapes_num - 1; j > 1; --j) {
      if (shape_infos[i].transforms[j] == "Padding") {
        continue;
      }
      resize_w = static_cast<float>(shape_infos[i].shapes[j][0]);
      break;
    }
    float scale = resize_w / origin_w;
    float im_info_data[] = {static_cast<float>(h), static_cast<float>(w),
                            scale};
    float im_shape_data[] = {origin_h, origin_w, 1.0};
    memcpy(im.data.data() + i * sample_shape * sizeof(float), imgs[i].data,
           sample_shape * sizeof(float));
    memcpy(im_info.data.data() + i * 3 * sizeof(float), im_info_data,
           3 * sizeof(float));
    memcpy(im_shape.data.data() + i * 3 * sizeof(float), im_shape_data,
           3 * sizeof(float));
  }

  inputs->clear();
  inputs->push_back(std::move(im));
  inputs->push_back(std::move(im_info));
  inputs->push_back(std::move(im_shape));
  return true;
}

bool DetPreprocess::Run(std::vector<cv::Mat>* imgs,
                        std::vector<DataBlob>* inputs,
                        std::vector<ShapeInfo>* shape_infos, int thread_num) {
  if ((*imgs).size() == 0) {
    std::cerr << "empty input image on DetPreprocess" << std::endl;
    return true;
  }
  if (!ShapeInfer(*imgs, shape_infos, thread_num)) {
    std::cerr << "ShapeInfer failed while call DetPreprocess::Run" << std::endl;
    return false;
  }
  if (!PrepareInputs(*shape_infos, imgs, inputs, thread_num)) {
    std::cerr << "PrepareInputs failed while call "
              << "DetPreprocess::PrepareInputs" << std::endl;
    return false;
  }
  return true;
}

}  //  namespace PaddleDeploy
