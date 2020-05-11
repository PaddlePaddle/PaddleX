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

namespace PaddleX {

void Model::create_predictor(const std::string& model_dir,
                             bool use_gpu,
                             bool use_trt,
                             int gpu_id) {
  // 读取配置文件
  if (!load_config(model_dir)) {
    std::cerr << "Parse file 'model.yml' failed!" << std::endl;
    exit(-1);
  }
  paddle::AnalysisConfig config;
  std::string model_file = model_dir + OS_PATH_SEP + "__model__";
  std::string params_file = model_dir + OS_PATH_SEP + "__params__";
  config.SetModel(model_file, params_file);
  if (use_gpu) {
    config.EnableUseGpu(100, gpu_id);
  } else {
    config.DisableGpu();
  }
  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  // 开启内存优化
  config.EnableMemoryOptim();
  if (use_trt) {
    config.EnableTensorRtEngine(
        1 << 20 /* workspace_size*/,
        32 /* max_batch_size*/,
        20 /* min_subgraph_size*/,
        paddle::AnalysisConfig::Precision::kFloat32 /* precision*/,
        true /* use_static*/,
        false /* use_calib_mode*/);
  }
  predictor_ = std::move(CreatePaddlePredictor(config));
}

bool Model::load_config(const std::string& model_dir) {
  std::string yaml_file = model_dir + OS_PATH_SEP + "model.yml";
  YAML::Node config = YAML::LoadFile(yaml_file);
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
  // 读入label list
  labels.clear();
  for (const auto& item : config["_Attributes"]["labels"]) {
    int index = labels.size();
    labels[index] = item.as<std::string>();
  }
  return true;
}

bool Model::preprocess(const cv::Mat& input_im, ImageBlob* blob) {
  cv::Mat im = input_im.clone();
  if (!transforms_.Run(&im, &inputs_)) {
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
  if (!preprocess(im, &inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }
  // 使用加载的模型进行预测
  auto in_tensor = predictor_->GetInputTensor("image");
  int h = inputs_.new_im_size_[0];
  int w = inputs_.new_im_size_[1];
  in_tensor->Reshape({1, 3, h, w});
  in_tensor->copy_from_cpu(inputs_.im_data_.data());
  predictor_->ZeroCopyRun();
  // 取出模型的输出结果
  auto output_names = predictor_->GetOutputNames();
  auto output_tensor = predictor_->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_tensor->shape();
  int size = 1;
  for (const auto& i : output_shape) {
    size *= i;
  }
  outputs_.resize(size);
  output_tensor->copy_to_cpu(outputs_.data());
  // 对模型输出结果进行后处理
  auto ptr = std::max_element(std::begin(outputs_), std::end(outputs_));
  result->category_id = std::distance(std::begin(outputs_), ptr);
  result->score = *ptr;
  result->category = labels[result->category_id];
}

bool Model::predict(const cv::Mat& im, DetResult* result) {
  result->clear();
  inputs_.clear();
  if (type == "classifier") {
    std::cerr << "Loading model is a 'classifier', ClsResult should be passed "
                 "to function predict()!"
              << std::endl;
    return false;
  } else if (type == "segmenter") {
    std::cerr << "Loading model is a 'segmenter', SegResult should be passed "
                 "to function predict()!"
              << std::endl;
    return false;
  }

  // 处理输入图像
  if (!preprocess(im, &inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }

  int h = inputs_.new_im_size_[0];
  int w = inputs_.new_im_size_[1];
  auto im_tensor = predictor_->GetInputTensor("image");
  im_tensor->Reshape({1, 3, h, w});
  im_tensor->copy_from_cpu(inputs_.im_data_.data());
  if (name == "YOLOv3") {
    auto im_size_tensor = predictor_->GetInputTensor("im_size");
    im_size_tensor->Reshape({1, 2});
    im_size_tensor->copy_from_cpu(inputs_.ori_im_size_.data());
  } else if (name == "FasterRCNN" || name == "MaskRCNN") {
    auto im_info_tensor = predictor_->GetInputTensor("im_info");
    auto im_shape_tensor = predictor_->GetInputTensor("im_shape");
    im_info_tensor->Reshape({1, 3});
    im_shape_tensor->Reshape({1, 3});
    float ori_h = static_cast<float>(inputs_.ori_im_size_[0]);
    float ori_w = static_cast<float>(inputs_.ori_im_size_[1]);
    float new_h = static_cast<float>(inputs_.new_im_size_[0]);
    float new_w = static_cast<float>(inputs_.new_im_size_[1]);
    float im_info[] = {new_h, new_w, inputs_.scale};
    float im_shape[] = {ori_h, ori_w, 1.0};
    im_info_tensor->copy_from_cpu(im_info);
    im_shape_tensor->copy_from_cpu(im_shape);
  }
  // 使用加载的模型进行预测
  predictor_->ZeroCopyRun();

  std::vector<float> output_box;
  auto output_names = predictor_->GetOutputNames();
  auto output_box_tensor = predictor_->GetOutputTensor(output_names[0]);
  std::vector<int> output_box_shape = output_box_tensor->shape();
  int size = 1;
  for (const auto& i : output_box_shape) {
    size *= i;
  }
  output_box.resize(size);
  output_box_tensor->copy_to_cpu(output_box.data());
  if (size < 6) {
    std::cerr << "[WARNING] There's no object detected." << std::endl;
    return true;
  }
  int num_boxes = size / 6;
  // 解析预测框box
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
  // 实例分割需解析mask
  if (name == "MaskRCNN") {
    std::vector<float> output_mask;
    auto output_mask_tensor = predictor_->GetOutputTensor(output_names[1]);
    std::vector<int> output_mask_shape = output_mask_tensor->shape();
    int masks_size = 1;
    for (const auto& i : output_mask_shape) {
      masks_size *= i;
    }
    int mask_pixels = output_mask_shape[2] * output_mask_shape[3];
    int classes = output_mask_shape[1];
    output_mask.resize(masks_size);
    output_mask_tensor->copy_to_cpu(output_mask.data());
    result->mask_resolution = output_mask_shape[2];
    for (int i = 0; i < result->boxes.size(); ++i) {
      Box* box = &result->boxes[i];
      auto begin_mask =
          output_mask.begin() + (i * classes + box->category_id) * mask_pixels;
      auto end_mask = begin_mask + mask_pixels;
      box->mask.data.assign(begin_mask, end_mask);
      box->mask.shape = {static_cast<int>(box->coordinate[2]),
                         static_cast<int>(box->coordinate[3])};
    }
  }
}

bool Model::predict(const cv::Mat& im, SegResult* result) {
  result->clear();
  inputs_.clear();
  if (type == "classifier") {
    std::cerr << "Loading model is a 'classifier', ClsResult should be passed "
                 "to function predict()!"
              << std::endl;
    return false;
  } else if (type == "detector") {
    std::cerr << "Loading model is a 'detector', DetResult should be passed to "
                 "function predict()!"
              << std::endl;
    return false;
  }

  // 处理输入图像
  if (!preprocess(im, &inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }

  int h = inputs_.new_im_size_[0];
  int w = inputs_.new_im_size_[1];
  auto im_tensor = predictor_->GetInputTensor("image");
  im_tensor->Reshape({1, 3, h, w});
  im_tensor->copy_from_cpu(inputs_.im_data_.data());

  // 使用加载的模型进行预测
  predictor_->ZeroCopyRun();

  // 获取预测置信度，经过argmax后的labelmap
  auto output_names = predictor_->GetOutputNames();
  auto output_label_tensor = predictor_->GetOutputTensor(output_names[0]);
  std::vector<int> output_label_shape = output_label_tensor->shape();
  int size = 1;
  for (const auto& i : output_label_shape) {
    size *= i;
    result->label_map.shape.push_back(i);
  }
  result->label_map.data.resize(size);
  output_label_tensor->copy_to_cpu(result->label_map.data.data());

  // 获取预测置信度scoremap
  auto output_score_tensor = predictor_->GetOutputTensor(output_names[1]);
  std::vector<int> output_score_shape = output_score_tensor->shape();
  size = 1;
  for (const auto& i : output_score_shape) {
    size *= i;
    result->score_map.shape.push_back(i);
  }
  result->score_map.data.resize(size);
  output_score_tensor->copy_to_cpu(result->score_map.data.data());

  // 解析输出结果到原图大小
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
      mask_label = mask_label(cv::Rect(0, 0, padding_w, padding_h));
      mask_score = mask_score(cv::Rect(0, 0, padding_w, padding_h));
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
                 cv::INTER_NEAREST);
    }
    ++idx;
  }
  result->label_map.data.assign(mask_label.begin<uint8_t>(),
                                mask_label.end<uint8_t>());
  result->label_map.shape = {mask_label.rows, mask_label.cols};
  result->score_map.data.assign(mask_score.begin<float>(),
                                mask_score.end<float>());
  result->score_map.shape = {mask_score.rows, mask_score.cols};
}

}  // namespce of PaddleX
