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


namespace PaddleX {

void Model::create_predictor(const std::string& model_dir,
                            const std::string& cfg_file,
                            std::string device) {
    InferenceEngine::Core ie;
    network_ = ie.ReadNetwork(
      model_dir, model_dir.substr(0, model_dir.size() - 4) + ".bin");
    network_.setBatchSize(1);

    InferenceEngine::InputsDataMap inputInfo(network_.getInputsInfo());
    std::string imageInputName;
    for (const auto & inputInfoItem : inputInfo) {
      if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {
        imageInputName = inputInfoItem.first;
        inputInfoItem.second->setPrecision(InferenceEngine::Precision::FP32);
        inputInfoItem.second->getPreProcess().setResizeAlgorithm(
          InferenceEngine::RESIZE_BILINEAR);
        inputInfoItem.second->setLayout(InferenceEngine::Layout::NCHW);
      }
      if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {
        imageInputName = inputInfoItem.first;
        inputInfoItem.second->setPrecision(InferenceEngine::Precision::FP32);
      }
    }
    if (device == "MYRIAD") {
      std::map<std::string, std::string> networkConfig;
      networkConfig["VPU_HW_STAGES_OPTIMIZATION"] = "ON";
      executable_network_ = ie.LoadNetwork(network_, device, networkConfig);
    } else {
      executable_network_ = ie.LoadNetwork(network_, device);
    }
    load_config(cfg_file);
}

bool Model::load_config(const std::string& cfg_file) {
  YAML::Node config = YAML::LoadFile(cfg_file);
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
  // init preprocess ops
  transforms_.Init(config["Transforms"], type, to_rgb);
  // read label list
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
  // preprocess
  InferenceEngine::InferRequest infer_request =
    executable_network_.CreateInferRequest();
  std::string input_name = network_.getInputsInfo().begin()->first;
  inputs_.blob = infer_request.GetBlob(input_name);
  cv::Mat im_clone = im.clone();
  if (!preprocess(&im_clone, &inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }

  // predict
  infer_request.Infer();

  std::string output_name = network_.getOutputsInfo().begin()->first;
  output_ = infer_request.GetBlob(output_name);
  InferenceEngine::MemoryBlob::CPtr moutput =
    InferenceEngine::as<InferenceEngine::MemoryBlob>(output_);
  InferenceEngine::TensorDesc blob_output = moutput->getTensorDesc();
  std::vector<size_t> output_shape = blob_output.getDims();
  auto moutputHolder = moutput->rmap();
  float* outputs_data = moutputHolder.as<float *>();
  int size = 1;
  for (auto& i : output_shape) {
    size *= static_cast<int>(i);
  }
  // post process
  auto ptr = std::max_element(outputs_data, outputs_data + size);
  result->category_id = std::distance(outputs_data, ptr);
  result->score = *ptr;
  result->category = labels[result->category_id];
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
  InferenceEngine::InferRequest infer_request =
    executable_network_.CreateInferRequest();
  InferenceEngine::InputsDataMap input_maps = network_.getInputsInfo();
  std::string inputName;
  for (const auto & input_map : input_maps) {
    if (input_map.second->getTensorDesc().getDims().size() == 4) {
      inputName = input_map.first;
      inputs_.blob = infer_request.GetBlob(inputName);
    }
    if (input_map.second->getTensorDesc().getDims().size() == 2) {
      inputName = input_map.first;
      inputs_.ori_im_size_ = infer_request.GetBlob(inputName);
    }
  }
  cv::Mat im_clone = im.clone();
  if (!preprocess(&im_clone, &inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }

  infer_request.Infer();

  InferenceEngine::OutputsDataMap out_maps = network_.getOutputsInfo();
  std::string outputName;
  for (const auto & output_map : out_maps) {
    if (output_map.second->getTensorDesc().getDims().size() == 2) {
      outputName = output_map.first;
    }
  }
  if (outputName.empty()) {
    std::cerr << "get result node failed!" << std::endl;
    return false;
  }
  InferenceEngine::Blob::Ptr output = infer_request.GetBlob(outputName);
  InferenceEngine::MemoryBlob::CPtr moutput =
    InferenceEngine::as<InferenceEngine::MemoryBlob>(output);
  InferenceEngine::TensorDesc blob_output = moutput->getTensorDesc();
  std::vector<size_t> output_shape = blob_output.getDims();
  auto moutputHolder = moutput->rmap();
  float* data = moutputHolder.as<float *>();
  int size = 1;
  for (auto& i : output_shape) {
    size *= static_cast<int>(i);
  }
  int num_boxes = size / 6;
  for (int i = 0; i < num_boxes; ++i) {
    if (data[i * 6] >= 0) {
      Box box;
      box.category_id = static_cast<int>(data[i * 6]);
      box.category = labels[box.category_id];
      box.score = data[i * 6 + 1];
      float xmin = data[i * 6 + 2];
      float ymin = data[i * 6 + 3];
      float xmax = data[i * 6 + 4];
      float ymax = data[i * 6 + 5];
      float w = xmax - xmin + 1;
      float h = ymax - ymin + 1;
      box.coordinate = {xmin, ymin, w, h};
      result->boxes.push_back(std::move(box));
    }
  }
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
  // init infer
  InferenceEngine::InferRequest infer_request =
    executable_network_.CreateInferRequest();
  std::string input_name = network_.getInputsInfo().begin()->first;
  inputs_.blob = infer_request.GetBlob(input_name);

  // preprocess
  cv::Mat im_clone = im.clone();
  if (!preprocess(&im_clone, &inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }

  // predict
  infer_request.Infer();

  InferenceEngine::OutputsDataMap out_map = network_.getOutputsInfo();
  auto iter = out_map.begin();
  iter++;

  std::string output_name_label = iter->first;
  InferenceEngine::Blob::Ptr output_label =
    infer_request.GetBlob(output_name_label);
  InferenceEngine::MemoryBlob::CPtr moutput_label =
    InferenceEngine::as<InferenceEngine::MemoryBlob>(output_label);
  InferenceEngine::TensorDesc blob_label = moutput_label->getTensorDesc();
  std::vector<size_t> output_label_shape = blob_label.getDims();
  int size = 1;
  for (auto& i : output_label_shape) {
    size *= static_cast<int>(i);
    result->label_map.shape.push_back(static_cast<int>(i));
  }
  result->label_map.data.resize(size);
  auto moutputHolder_label = moutput_label->rmap();
  int* label_data = moutputHolder_label.as<int *>();
  memcpy(result->label_map.data.data(), label_data, moutput_label->byteSize());

  iter++;
  std::string output_name_score = iter->first;
  InferenceEngine::Blob::Ptr output_score =
    infer_request.GetBlob(output_name_score);
  InferenceEngine::MemoryBlob::CPtr moutput_score =
    InferenceEngine::as<InferenceEngine::MemoryBlob>(output_score);
  InferenceEngine::TensorDesc blob_score = moutput_score->getTensorDesc();
  std::vector<size_t> output_score_shape = blob_score.getDims();
  size = 1;
  for (auto& i : output_score_shape) {
    size *= static_cast<int>(i);
    result->score_map.shape.push_back(static_cast<int>(i));
  }
  result->score_map.data.resize(size);
  auto moutputHolder_score = moutput_score->rmap();
  float* score_data = moutputHolder_score.as<float *>();
  memcpy(result->score_map.data.data(), score_data, moutput_score->byteSize());

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
      auto padding_h = before_shape[0];
      auto padding_w = before_shape[1];
      mask_label = mask_label(cv::Rect(0, 0, padding_h, padding_w));
      mask_score = mask_score(cv::Rect(0, 0, padding_h, padding_w));
    } else if (*iter == "resize") {
      auto before_shape = inputs_.im_size_before_resize_[len_postprocess - idx];
      inputs_.im_size_before_resize_.pop_back();
      auto resize_h = before_shape[0];
      auto resize_w = before_shape[1];
      cv::resize(mask_label,
                 mask_label,
                 cv::Size(resize_w, resize_h),
                 0,
                 0,
                 cv::INTER_NEAREST);
      cv::resize(mask_score,
                 mask_score,
                 cv::Size(resize_w, resize_h),
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
}  // namespace PaddleX
