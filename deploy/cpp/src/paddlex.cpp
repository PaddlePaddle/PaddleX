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

#include <math.h>
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <cstring>
#include "include/paddlex/paddlex.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace PaddleX {

void Model::create_predictor(const std::string& model_dir,
                             bool use_gpu,
                             bool use_trt,
                             bool use_mkl,
                             int mkl_thread_num,
                             int gpu_id,
                             std::string key,
                             bool use_ir_optim) {
  paddle::AnalysisConfig config;
  std::string model_file = model_dir + OS_PATH_SEP + "__model__";
  std::string params_file = model_dir + OS_PATH_SEP + "__params__";
  std::string yaml_file = model_dir + OS_PATH_SEP + "model.yml";
  std::string yaml_input = "";
#ifdef WITH_ENCRYPTION
  if (key != "") {
    model_file = model_dir + OS_PATH_SEP + "__model__.encrypted";
    params_file = model_dir + OS_PATH_SEP + "__params__.encrypted";
    yaml_file = model_dir + OS_PATH_SEP + "model.yml.encrypted";
    paddle_security_load_model(
        &config, key.c_str(), model_file.c_str(), params_file.c_str());
    yaml_input = decrypt_file(yaml_file.c_str(), key.c_str());
  }
#endif
  if (yaml_input == "") {
    // read yaml file
    std::ifstream yaml_fin(yaml_file);
    yaml_fin.seekg(0, std::ios::end);
    size_t yaml_file_size = yaml_fin.tellg();
    yaml_input.assign(yaml_file_size, ' ');
    yaml_fin.seekg(0);
    yaml_fin.read(&yaml_input[0], yaml_file_size);
  }
  // load yaml file
  if (!load_config(yaml_input)) {
    std::cerr << "Parse file 'model.yml' failed!" << std::endl;
    exit(-1);
  }

  if (key == "") {
    config.SetModel(model_file, params_file);
  }
  if (use_mkl && !use_gpu) {
    if (name != "HRNet" && name != "DeepLabv3p" && name != "PPYOLO") {
        config.EnableMKLDNN();
        config.SetCpuMathLibraryNumThreads(mkl_thread_num);
    } else {
        std::cerr << "HRNet/DeepLabv3p/PPYOLO are not supported "
                  << "for the use of mkldnn" << std::endl;
    }
  }
  if (use_gpu) {
    config.EnableUseGpu(100, gpu_id);
  } else {
    config.DisableGpu();
  }
  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  // enable graph Optim
#if defined(__arm__) || defined(__aarch64__)
  config.SwitchIrOptim(false);
#else
  config.SwitchIrOptim(use_ir_optim);
#endif
  // enable Memory Optim
  config.EnableMemoryOptim();
  if (use_trt && use_gpu) {
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

bool Model::load_config(const std::string& yaml_input) {
  YAML::Node config = YAML::Load(yaml_input);
  type = config["_Attributes"]["model_type"].as<std::string>();
  name = config["Model"].as<std::string>();
  std::string version = config["version"].as<std::string>();
  if (version[0] == '0') {
    std::cerr << "[Init] Version of the loaded model is lower than 1.0.0, "
              << "deployment cannot be done, please refer to "
              << "https://github.com/PaddlePaddle/PaddleX/blob/develop/docs"
              << "/tutorials/deploy/upgrade_version.md "
              << "to transfer version." << std::endl;
    return false;
  }
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
  // build data preprocess stream
  transforms_.Init(config["Transforms"], to_rgb);
  // read label list
  labels.clear();
  for (const auto& item : config["_Attributes"]["labels"]) {
    int index = labels.size();
    labels[index] = item.as<std::string>();
  }
  if (config["_init_params"]["input_channel"].IsDefined()) {
    input_channel_ = config["_init_params"]["input_channel"].as<int>();
  } else {
    input_channel_ = 3;
  }
  return true;
}

bool Model::preprocess(const cv::Mat& input_im, ImageBlob* blob) {
  cv::Mat im = input_im.clone();
  if (!transforms_.Run(&im, blob)) {
    return false;
  }
  return true;
}

// use openmp
bool Model::preprocess(const std::vector<cv::Mat>& input_im_batch,
                       std::vector<ImageBlob>* blob_batch,
                       int thread_num) {
  int batch_size = input_im_batch.size();
  bool success = true;
  thread_num = std::min(thread_num, batch_size);
  #pragma omp parallel for num_threads(thread_num)
  for (int i = 0; i < input_im_batch.size(); ++i) {
    cv::Mat im = input_im_batch[i].clone();
    if (!transforms_.Run(&im, &(*blob_batch)[i])) {
      success = false;
    }
  }
  return success;
}

bool Model::predict(const cv::Mat& im, ClsResult* result) {
  inputs_.clear();
  if (type == "detector") {
    std::cerr << "Loading model is a 'detector', DetResult should be passed to "
                 "function predict()!" << std::endl;
    return false;
  } else if (type == "segmenter") {
    std::cerr << "Loading model is a 'segmenter', SegResult should be passed "
                 "to function predict()!" << std::endl;
    return false;
  }
  // im preprocess
  if (!preprocess(im, &inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }
  // predict
  auto in_tensor = predictor_->GetInputTensor("image");
  int h = inputs_.new_im_size_[0];
  int w = inputs_.new_im_size_[1];
  in_tensor->Reshape({1, input_channel_, h, w});
  in_tensor->copy_from_cpu(inputs_.im_data_.data());
  predictor_->ZeroCopyRun();
  // get result
  auto output_names = predictor_->GetOutputNames();
  auto output_tensor = predictor_->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_tensor->shape();
  int size = 1;
  for (const auto& i : output_shape) {
    size *= i;
  }
  outputs_.resize(size);
  output_tensor->copy_to_cpu(outputs_.data());
  // postprocess
  auto ptr = std::max_element(std::begin(outputs_), std::end(outputs_));
  result->category_id = std::distance(std::begin(outputs_), ptr);
  result->score = *ptr;
  result->category = labels[result->category_id];
  return true;
}

bool Model::predict(const std::vector<cv::Mat>& im_batch,
                    std::vector<ClsResult>* results,
                    int thread_num) {
  for (auto& inputs : inputs_batch_) {
    inputs.clear();
  }
  if (type == "detector") {
    std::cerr << "Loading model is a 'detector', DetResult should be passed to "
                 "function predict()!" << std::endl;
    return false;
  } else if (type == "segmenter") {
    std::cerr << "Loading model is a 'segmenter', SegResult should be passed "
                 "to function predict()!" << std::endl;
    return false;
  }
  inputs_batch_.assign(im_batch.size(), ImageBlob());
  // preprocess
  if (!preprocess(im_batch, &inputs_batch_, thread_num)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }
  // predict
  int batch_size = im_batch.size();
  auto in_tensor = predictor_->GetInputTensor("image");
  int h = inputs_batch_[0].new_im_size_[0];
  int w = inputs_batch_[0].new_im_size_[1];
  in_tensor->Reshape({batch_size, input_channel_, h, w});
  std::vector<float> inputs_data(batch_size * input_channel_ * h * w);
  for (int i = 0; i < batch_size; ++i) {
    std::copy(inputs_batch_[i].im_data_.begin(),
              inputs_batch_[i].im_data_.end(),
              inputs_data.begin() + i * input_channel_ * h * w);
  }
  in_tensor->copy_from_cpu(inputs_data.data());
  // in_tensor->copy_from_cpu(inputs_.im_data_.data());
  predictor_->ZeroCopyRun();
  // get result
  auto output_names = predictor_->GetOutputNames();
  auto output_tensor = predictor_->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_tensor->shape();
  int size = 1;
  for (const auto& i : output_shape) {
    size *= i;
  }
  outputs_.resize(size);
  output_tensor->copy_to_cpu(outputs_.data());
  // postprocess
  (*results).clear();
  (*results).resize(batch_size);
  int single_batch_size = size / batch_size;
  for (int i = 0; i < batch_size; ++i) {
    auto start_ptr = std::begin(outputs_);
    auto end_ptr = std::begin(outputs_);
    std::advance(start_ptr, i * single_batch_size);
    std::advance(end_ptr, (i + 1) * single_batch_size);
    auto ptr = std::max_element(start_ptr, end_ptr);
    (*results)[i].category_id = std::distance(start_ptr, ptr);
    (*results)[i].score = *ptr;
    (*results)[i].category = labels[(*results)[i].category_id];
  }
  return true;
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

  // preprocess
  if (!preprocess(im, &inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }

  int h = inputs_.new_im_size_[0];
  int w = inputs_.new_im_size_[1];
  auto im_tensor = predictor_->GetInputTensor("image");
  im_tensor->Reshape({1, input_channel_, h, w});
  im_tensor->copy_from_cpu(inputs_.im_data_.data());

  if (name == "YOLOv3" || name == "PPYOLO") {
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
  // predict
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
  // box postprocess
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
  // mask postprocess
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
      box->mask.shape = {static_cast<int>(box->coordinate[2]),
                         static_cast<int>(box->coordinate[3])};
      auto begin_mask =
          output_mask.data() + (i * classes + box->category_id) * mask_pixels;
      cv::Mat bin_mask(result->mask_resolution,
                     result->mask_resolution,
                     CV_32FC1,
                     begin_mask);
      cv::resize(bin_mask,
               bin_mask,
               cv::Size(box->mask.shape[0], box->mask.shape[1]));
      cv::threshold(bin_mask, bin_mask, 0.5, 1, cv::THRESH_BINARY);
      auto mask_int_begin = reinterpret_cast<float*>(bin_mask.data);
      auto mask_int_end =
        mask_int_begin + box->mask.shape[0] * box->mask.shape[1];
      box->mask.data.assign(mask_int_begin, mask_int_end);
    }
  }
  return true;
}

bool Model::predict(const std::vector<cv::Mat>& im_batch,
                    std::vector<DetResult>* results,
                    int thread_num) {
  for (auto& inputs : inputs_batch_) {
    inputs.clear();
  }
  if (type == "classifier") {
    std::cerr << "Loading model is a 'classifier', ClsResult should be passed "
                 "to function predict()!" << std::endl;
    return false;
  } else if (type == "segmenter") {
    std::cerr << "Loading model is a 'segmenter', SegResult should be passed "
                 "to function predict()!" << std::endl;
    return false;
  }

  inputs_batch_.assign(im_batch.size(), ImageBlob());
  int batch_size = im_batch.size();
  // preprocess
  if (!preprocess(im_batch, &inputs_batch_, thread_num)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }
  // RCNN model padding
  if (batch_size > 1) {
    if (name == "FasterRCNN" || name == "MaskRCNN") {
      int max_h = -1;
      int max_w = -1;
      for (int i = 0; i < batch_size; ++i) {
        max_h = std::max(max_h, inputs_batch_[i].new_im_size_[0]);
        max_w = std::max(max_w, inputs_batch_[i].new_im_size_[1]);
        // std::cout << "(" << inputs_batch_[i].new_im_size_[0]
        //          << ", " << inputs_batch_[i].new_im_size_[1]
        //          <<  ")" << std::endl;
      }
      thread_num = std::min(thread_num, batch_size);
      #pragma omp parallel for num_threads(thread_num)
      for (int i = 0; i < batch_size; ++i) {
        int h = inputs_batch_[i].new_im_size_[0];
        int w = inputs_batch_[i].new_im_size_[1];
        int c = im_batch[i].channels();
        if (max_h != h || max_w != w) {
          std::vector<float> temp_buffer(c * max_h * max_w);
          float* temp_ptr = temp_buffer.data();
          float* ptr = inputs_batch_[i].im_data_.data();
          for (int cur_channel = c - 1; cur_channel >= 0; --cur_channel) {
            int ori_pos = cur_channel * h * w + (h - 1) * w;
            int des_pos = cur_channel * max_h * max_w + (h - 1) * max_w;
            int last_pos = cur_channel * h * w;
            for (; ori_pos >= last_pos; ori_pos -= w, des_pos -= max_w) {
              memcpy(temp_ptr + des_pos, ptr + ori_pos, w * sizeof(float));
            }
          }
          inputs_batch_[i].im_data_.swap(temp_buffer);
          inputs_batch_[i].new_im_size_[0] = max_h;
          inputs_batch_[i].new_im_size_[1] = max_w;
        }
      }
    }
  }
  int h = inputs_batch_[0].new_im_size_[0];
  int w = inputs_batch_[0].new_im_size_[1];
  auto im_tensor = predictor_->GetInputTensor("image");
  im_tensor->Reshape({batch_size, input_channel_, h, w});
  std::vector<float> inputs_data(batch_size * input_channel_ * h * w);
  for (int i = 0; i < batch_size; ++i) {
    std::copy(inputs_batch_[i].im_data_.begin(),
              inputs_batch_[i].im_data_.end(),
              inputs_data.begin() + i * input_channel_ * h * w);
  }
  im_tensor->copy_from_cpu(inputs_data.data());
  if (name == "YOLOv3" || name == "PPYOLO") {
    auto im_size_tensor = predictor_->GetInputTensor("im_size");
    im_size_tensor->Reshape({batch_size, 2});
    std::vector<int> inputs_data_size(batch_size * 2);
    for (int i = 0; i < batch_size; ++i) {
      std::copy(inputs_batch_[i].ori_im_size_.begin(),
                inputs_batch_[i].ori_im_size_.end(),
                inputs_data_size.begin() + 2 * i);
    }
    im_size_tensor->copy_from_cpu(inputs_data_size.data());
  } else if (name == "FasterRCNN" || name == "MaskRCNN") {
    auto im_info_tensor = predictor_->GetInputTensor("im_info");
    auto im_shape_tensor = predictor_->GetInputTensor("im_shape");
    im_info_tensor->Reshape({batch_size, 3});
    im_shape_tensor->Reshape({batch_size, 3});

    std::vector<float> im_info(3 * batch_size);
    std::vector<float> im_shape(3 * batch_size);
    for (int i = 0; i < batch_size; ++i) {
      float ori_h = static_cast<float>(inputs_batch_[i].ori_im_size_[0]);
      float ori_w = static_cast<float>(inputs_batch_[i].ori_im_size_[1]);
      float new_h = static_cast<float>(inputs_batch_[i].new_im_size_[0]);
      float new_w = static_cast<float>(inputs_batch_[i].new_im_size_[1]);
      im_info[i * 3] = new_h;
      im_info[i * 3 + 1] = new_w;
      im_info[i * 3 + 2] = inputs_batch_[i].scale;
      im_shape[i * 3] = ori_h;
      im_shape[i * 3 + 1] = ori_w;
      im_shape[i * 3 + 2] = 1.0;
    }
    im_info_tensor->copy_from_cpu(im_info.data());
    im_shape_tensor->copy_from_cpu(im_shape.data());
  }
  // predict
  predictor_->ZeroCopyRun();

  // get all box
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
  auto lod_vector = output_box_tensor->lod();
  int num_boxes = size / 6;
  // box postprocess
  (*results).clear();
  (*results).resize(batch_size);
  for (int i = 0; i < lod_vector[0].size() - 1; ++i) {
    for (int j = lod_vector[0][i]; j < lod_vector[0][i + 1]; ++j) {
      Box box;
      box.category_id = static_cast<int>(round(output_box[j * 6]));
      box.category = labels[box.category_id];
      box.score = output_box[j * 6 + 1];
      float xmin = output_box[j * 6 + 2];
      float ymin = output_box[j * 6 + 3];
      float xmax = output_box[j * 6 + 4];
      float ymax = output_box[j * 6 + 5];
      float w = xmax - xmin + 1;
      float h = ymax - ymin + 1;
      box.coordinate = {xmin, ymin, w, h};
      (*results)[i].boxes.push_back(std::move(box));
    }
  }

  // mask postprocess
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
    int mask_idx = 0;
    for (int i = 0; i < lod_vector[0].size() - 1; ++i) {
      (*results)[i].mask_resolution = output_mask_shape[2];
      for (int j = 0; j < (*results)[i].boxes.size(); ++j) {
        Box* box = &(*results)[i].boxes[i];
        int category_id = box->category_id;
        box->mask.shape = {static_cast<int>(box->coordinate[2]),
                          static_cast<int>(box->coordinate[3])};
        auto begin_mask =
          output_mask.data() + (i * classes + box->category_id) * mask_pixels;
        cv::Mat bin_mask(output_mask_shape[2],
                      output_mask_shape[2],
                      CV_32FC1,
                      begin_mask);
        cv::resize(bin_mask,
                bin_mask,
                cv::Size(box->mask.shape[0], box->mask.shape[1]));
        cv::threshold(bin_mask, bin_mask, 0.5, 1, cv::THRESH_BINARY);
        auto mask_int_begin = reinterpret_cast<float*>(bin_mask.data);
        auto mask_int_end =
          mask_int_begin + box->mask.shape[0] * box->mask.shape[1];
        box->mask.data.assign(mask_int_begin, mask_int_end);
        mask_idx++;
      }
    }
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

  // preprocess
  if (!preprocess(im, &inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }

  int h = inputs_.new_im_size_[0];
  int w = inputs_.new_im_size_[1];
  auto im_tensor = predictor_->GetInputTensor("image");
  im_tensor->Reshape({1, input_channel_, h, w});
  im_tensor->copy_from_cpu(inputs_.im_data_.data());

  // predict
  predictor_->ZeroCopyRun();

  // get labelmap
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

  // get scoremap
  auto output_score_tensor = predictor_->GetOutputTensor(output_names[1]);
  std::vector<int> output_score_shape = output_score_tensor->shape();
  size = 1;
  for (const auto& i : output_score_shape) {
    size *= i;
    result->score_map.shape.push_back(i);
  }

  result->score_map.data.resize(size);
  output_score_tensor->copy_to_cpu(result->score_map.data.data());

  // get origin image result
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

bool Model::predict(const std::vector<cv::Mat>& im_batch,
                    std::vector<SegResult>* results,
                    int thread_num) {
  for (auto& inputs : inputs_batch_) {
    inputs.clear();
  }
  if (type == "classifier") {
    std::cerr << "Loading model is a 'classifier', ClsResult should be passed "
                 "to function predict()!" << std::endl;
    return false;
  } else if (type == "detector") {
    std::cerr << "Loading model is a 'detector', DetResult should be passed to "
                 "function predict()!" << std::endl;
    return false;
  }

  // preprocess
  inputs_batch_.assign(im_batch.size(), ImageBlob());
  if (!preprocess(im_batch, &inputs_batch_, thread_num)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }

  int batch_size = im_batch.size();
  (*results).clear();
  (*results).resize(batch_size);
  int h = inputs_batch_[0].new_im_size_[0];
  int w = inputs_batch_[0].new_im_size_[1];
  auto im_tensor = predictor_->GetInputTensor("image");
  im_tensor->Reshape({batch_size, input_channel_, h, w});
  std::vector<float> inputs_data(batch_size * input_channel_ * h * w);
  for (int i = 0; i < batch_size; ++i) {
    std::copy(inputs_batch_[i].im_data_.begin(),
              inputs_batch_[i].im_data_.end(),
              inputs_data.begin() + i * input_channel_ * h * w);
  }
  im_tensor->copy_from_cpu(inputs_data.data());
  // im_tensor->copy_from_cpu(inputs_.im_data_.data());

  // predict
  predictor_->ZeroCopyRun();

  // get labelmap
  auto output_names = predictor_->GetOutputNames();
  auto output_label_tensor = predictor_->GetOutputTensor(output_names[0]);
  std::vector<int> output_label_shape = output_label_tensor->shape();
  int size = 1;
  for (const auto& i : output_label_shape) {
    size *= i;
  }

  std::vector<int64_t> output_labels(size, 0);
  output_label_tensor->copy_to_cpu(output_labels.data());
  auto output_labels_iter = output_labels.begin();

  int single_batch_size = size / batch_size;
  for (int i = 0; i < batch_size; ++i) {
    (*results)[i].label_map.data.resize(single_batch_size);
    (*results)[i].label_map.shape.push_back(1);
    for (int j = 1; j < output_label_shape.size(); ++j) {
      (*results)[i].label_map.shape.push_back(output_label_shape[j]);
    }
    std::copy(output_labels_iter + i * single_batch_size,
              output_labels_iter + (i + 1) * single_batch_size,
              (*results)[i].label_map.data.data());
  }

  // get scoremap
  auto output_score_tensor = predictor_->GetOutputTensor(output_names[1]);
  std::vector<int> output_score_shape = output_score_tensor->shape();
  size = 1;
  for (const auto& i : output_score_shape) {
    size *= i;
  }

  std::vector<float> output_scores(size, 0);
  output_score_tensor->copy_to_cpu(output_scores.data());
  auto output_scores_iter = output_scores.begin();

  int single_batch_score_size = size / batch_size;
  for (int i = 0; i < batch_size; ++i) {
    (*results)[i].score_map.data.resize(single_batch_score_size);
    (*results)[i].score_map.shape.push_back(1);
    for (int j = 1; j < output_score_shape.size(); ++j) {
      (*results)[i].score_map.shape.push_back(output_score_shape[j]);
    }
    std::copy(output_scores_iter + i * single_batch_score_size,
              output_scores_iter + (i + 1) * single_batch_score_size,
              (*results)[i].score_map.data.data());
  }

  // get origin image result
  for (int i = 0; i < batch_size; ++i) {
    std::vector<uint8_t> label_map((*results)[i].label_map.data.begin(),
                                   (*results)[i].label_map.data.end());
    cv::Mat mask_label((*results)[i].label_map.shape[1],
                       (*results)[i].label_map.shape[2],
                       CV_8UC1,
                       label_map.data());

    cv::Mat mask_score((*results)[i].score_map.shape[2],
                       (*results)[i].score_map.shape[3],
                       CV_32FC1,
                       (*results)[i].score_map.data.data());
    int idx = 1;
    int len_postprocess = inputs_batch_[i].im_size_before_resize_.size();
    for (std::vector<std::string>::reverse_iterator iter =
             inputs_batch_[i].reshape_order_.rbegin();
         iter != inputs_batch_[i].reshape_order_.rend();
         ++iter) {
      if (*iter == "padding") {
        auto before_shape =
            inputs_batch_[i].im_size_before_resize_[len_postprocess - idx];
        inputs_batch_[i].im_size_before_resize_.pop_back();
        auto padding_w = before_shape[0];
        auto padding_h = before_shape[1];
        mask_label = mask_label(cv::Rect(0, 0, padding_h, padding_w));
        mask_score = mask_score(cv::Rect(0, 0, padding_h, padding_w));
      } else if (*iter == "resize") {
        auto before_shape =
            inputs_batch_[i].im_size_before_resize_[len_postprocess - idx];
        inputs_batch_[i].im_size_before_resize_.pop_back();
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
    (*results)[i].label_map.data.assign(mask_label.begin<uint8_t>(),
                                       mask_label.end<uint8_t>());
    (*results)[i].label_map.shape = {mask_label.rows, mask_label.cols};
    (*results)[i].score_map.data.assign(mask_score.begin<float>(),
                                       mask_score.end<float>());
    (*results)[i].score_map.shape = {mask_score.rows, mask_score.cols};
  }
  return true;
}

}  // namespace PaddleX
