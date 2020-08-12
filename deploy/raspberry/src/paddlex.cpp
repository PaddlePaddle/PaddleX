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

#include "include/paddlex/paddlex.h"
#include <iostream>
#include <fstream>

using namespace paddle::lite_api;

namespace PaddleX {

void Model::create_predictor(const std::string& model_dir,
                            const std::string& cfg_dir,
                            int thread_num) {
  MobileConfig config;
  config.set_model_from_file(model_dir);
  config.set_threads(thread_num);  
  load_config(cfg_dir);
  predictor_ = CreatePaddlePredictor<MobileConfig>(config);
}

bool Model::load_config(const std::string& cfg_dir) {
  YAML::Node config = YAML::LoadFile(cfg_dir);
  type = config["_Attributes"]["model_type"].as<std::string>();
  name = config["Model"].as<std::string>();
  bool to_rgb = true;
  if (config["TransformsMode"].IsDefined()) {
    std::string mode = config["TransformsMode"].as<std::string>();
    if (mode == "BGR") {
      to_rgb = false;
    } else if (mode != "RGB") {
      std::cerr << "[Init] Only 'RGB' or 'BGR' is supported for TransformsMode"
                << std::endl;
      return false;
    }
  }
  // 构建数据处理流
  transforms_.Init(config["Transforms"], to_rgb);
  // 读入label lis
  for (const auto& item : config["_Attributes"]["labels"]) {
    int index = labels.size();
    labels[index] = item.as<std::string>();
  }

  return true;
}

bool Model::preprocess(cv::Mat* input_im, ImageBlob* inputs) {
  if (!transforms_.Run(input_im, inputs)) {
    return false;
  }
  return true;
}

bool Model::predict(const cv::Mat& im, ClsResult* result) {
  inputs_.clear();
  if (type == "detector") {
    std::cerr << "Loading model is a 'detector', DetResult should be passed to "
                 "function predict()!"
              << std::endl;
    return false;
  } else if (type == "segmenter") {
    std::cerr << "Loading model is a 'segmenter', SegResult should be passed "
                 "to function predict()!"
              << std::endl;
    return false;
  }
  // 处理输入图像
  inputs_.input_tensor_ = std::move(predictor_->GetInput(0));
  cv::Mat im_clone = im.clone();
  if (!preprocess(&im_clone, &inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }
  
  ;
  predictor_->Run();


  std::unique_ptr<const Tensor> output_tensor(std::move(predictor_->GetOutput(0)));
  const float *outputs_data = output_tensor->mutable_data<float>();


  // 对模型输出结果进行后处理
  auto ptr = std::max_element(outputs_data, outputs_data+sizeof(outputs_data));
  result->category_id = std::distance(outputs_data, ptr);
  result->score = *ptr;
  result->category = labels[result->category_id];
  //for (int i=0;i<sizeof(outputs_data);i++){
  //    std::cout <<  labels[i] << std::endl;
  //    std::cout <<  outputs_[i] << std::endl;
  //    }
}

bool Model::predict(const cv::Mat& im, DetResult* result) {
  inputs_.clear();
  result->clear();
  if (type == "classifier") {
    std::cerr << "Loading model is a 'classifier', ClsResult should be passed "
                 "to function predict()!" << std::endl;
    return false;
  } else if (type == "segmenter") {
    std::cerr << "Loading model is a 'segmenter', SegResult should be passed "
                 "to function predict()!" << std::endl;
    return false;
  }
  
  inputs_.input_tensor_ = std::move(predictor_->GetInput(0));

  cv::Mat im_clone = im.clone();
  if (!preprocess(&im_clone, &inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }
  int h = inputs_.new_im_size_[0];
  int w = inputs_.new_im_size_[1];
  if (name == "YOLOv3") {
    std::unique_ptr<Tensor> im_size_tensor(std::move(predictor_->GetInput(1)));
    const std::vector<int64_t> IM_SIZE_SHAPE = {1,2};
    im_size_tensor->Resize(IM_SIZE_SHAPE);
    auto *im_size_data = im_size_tensor->mutable_data<int>();
    memcpy(im_size_data, inputs_.ori_im_size_.data(), 1*2*sizeof(int));
  }
  
  
  predictor_->Run();
 
  

  auto output_names = predictor_->GetOutputNames();
  auto output_box_tensor = predictor_->GetTensor(output_names[0]);
  const float *output_box = output_box_tensor->mutable_data<float>();
  std::vector<int64_t> output_box_shape = output_box_tensor->shape();
  int size = 1;
  for (const auto& i : output_box_shape) {
    size *= i;
  }
  int num_boxes = size / 6;
  for (int i = 0; i < num_boxes; ++i) {
    Box box;
    box.category_id = static_cast<int>(round(output_box[i * 6]));
    box.category = labels[box.category_id];
    box.score = output_box[i * 6 + 1];
    float xmin = output_box[i * 6 + 2];
    float ymin = output_box[i * 6 + 3];
    float xmax = output_box[i * 6 + 4];
    float ymax = output_box[i * 6 + 5];
    float w = xmax - xmin + 1;
    float h = ymax - ymin + 1;
    box.coordinate = {xmin, ymin, w, h};
    result->boxes.push_back(std::move(box));
  }
  return true;
}


bool Model::predict(const cv::Mat& im, SegResult* result) {
  result->clear();
  inputs_.clear();
  if (type == "classifier") {
    std::cerr << "Loading model is a 'classifier', ClsResult should be passed "
                 "to function predict()!" << std::endl;
    return false;
  } else if (type == "detector") {
    std::cerr << "Loading model is a 'detector', DetResult should be passed to "
                 "function predict()!" << std::endl;
    return false;
  }
 
  inputs_.input_tensor_ = std::move(predictor_->GetInput(0));

  
  cv::Mat im_clone = im.clone();
  if (!preprocess(&im_clone, &inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }
  std::cout << "Preprocess is done" << std::endl;
 

  predictor_->Run();

 

  auto output_names = predictor_->GetOutputNames();

  auto output_label_tensor = predictor_->GetTensor(output_names[0]);
  std::cout << "output0" << output_names[0] << std::endl; 
  std::cout << "output1" << output_names[1] << std::endl; 
  const int64_t *label_data = output_label_tensor->mutable_data<int64_t>();
  std::vector<int64_t> output_label_shape = output_label_tensor->shape();
  int size = 1;
  for (const auto& i : output_label_shape) {
    size *= i;
    result->label_map.shape.push_back(i);
  }
  result->label_map.data.resize(size);
  memcpy(result->label_map.data.data(), label_data, size*sizeof(int64_t));

  auto output_score_tensor = predictor_->GetTensor(output_names[1]);
  const float *score_data = output_score_tensor->mutable_data<float>();
  std::vector<int64_t> output_score_shape = output_score_tensor->shape();
  size = 1;
  for (const auto& i : output_score_shape) {
    size *= i;
    result->score_map.shape.push_back(i);
  }
  result->score_map.data.resize(size);
  memcpy(result->score_map.data.data(), score_data, size*sizeof(float));


  std::vector<uint8_t> label_map(result->label_map.data.begin(),
                                 result->label_map.data.end());
  cv::Mat mask_label(result->label_map.shape[1],
                     result->label_map.shape[2],
                     CV_8UC1,
                     label_map.data());

  cv::Mat mask_score(result->score_map.shape[2],
                     result->score_map.shape[3],
                     CV_32FC1,
                     result->score_map.data.data());
  int idx = 1;
  int len_postprocess = inputs_.im_size_before_resize_.size();
  for (std::vector<std::string>::reverse_iterator iter =
           inputs_.reshape_order_.rbegin();
       iter != inputs_.reshape_order_.rend();
       ++iter) {
    if (*iter == "padding") {
      auto before_shape = inputs_.im_size_before_resize_[len_postprocess - idx];
      inputs_.im_size_before_resize_.pop_back();
      auto padding_w = before_shape[0];
      auto padding_h = before_shape[1];
      mask_label = mask_label(cv::Rect(0, 0, padding_h, padding_w));
      mask_score = mask_score(cv::Rect(0, 0, padding_h, padding_w));
    } else if (*iter == "resize") {
      auto before_shape = inputs_.im_size_before_resize_[len_postprocess - idx];
      inputs_.im_size_before_resize_.pop_back();
      auto resize_w = before_shape[0];
      auto resize_h = before_shape[1];
      cv::resize(mask_label,
                 mask_label,
                 cv::Size(resize_h, resize_w),
                 0,
                 0,
                 cv::INTER_NEAREST);
      cv::resize(mask_score,
                 mask_score,
                 cv::Size(resize_h, resize_w),
                 0,
                 0,
                 cv::INTER_LINEAR);
    }
    ++idx;
  }
  result->label_map.data.assign(mask_label.begin<uint8_t>(),
                                mask_label.end<uint8_t>());
  result->label_map.shape = {mask_label.rows, mask_label.cols};
  result->score_map.data.assign(mask_score.begin<float>(),
                                mask_score.end<float>());
  result->score_map.shape = {mask_score.rows, mask_score.cols};
  return true;
}
 
}  // namespce of PaddleX
