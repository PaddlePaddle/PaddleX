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

using namespace InferenceEngine;

namespace PaddleX {

void Model::create_predictor(const std::string& model_dir,
                            const std::string& cfg_dir,
                            std::string device) {
    Core ie;
    network_ = ie.ReadNetwork(model_dir, model_dir.substr(0, model_dir.size() - 4) + ".bin");
    network_.setBatchSize(1);
    InputInfo::Ptr input_info = network_.getInputsInfo().begin()->second;

    input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::FP32);

    load_config(cfg_dir);
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
  // 读入label list
  labels.clear();
  labels = config["_Attributes"]["labels"].as<std::vector<std::string>>();
  return true;
}

bool Model::preprocess(cv::Mat* input_im, ImageBlob* blob) {
  if (!transforms_.Run(input_im, &inputs_)) {
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

  executable_network = ie.LoadNetwork(network_, device);
  InferRequest infer_request = executable_network.CreateInferRequest();
  std::string input_name = network_.getInputsInfo().begin()->first;
  input_ = infer_request.GetBlob(input_name);

  auto im_clone = im.clone();
  if (!preprocess(&im_clone, inputs_)) {
    std::cerr << "Preprocess failed!" << std::endl;
    return false;
  }

  infer_request.Infer();

  std::string output_name = network_.getOutputsInfo().begin()->first;
  output_ = infer_request.GetBlob(output_name);
  MemoryBlob::CPtr moutput = as<MemoryBlob>(output);
  auto moutputHolder = moutput->rmap();
  float* outputs_data = moutputHolder.as<float *>();

  // 对模型输出结果进行后处理
  auto ptr = std::max_element(outputs_data, outputs_data+sizeof(outputs_));
  result->category_id = std::distance(outputs_data, ptr);
  result->score = *ptr;
  result->category = labels[result->category_id];
  //for (int i=0;i<sizeof(outputs_data);i++){
  //    std::cout <<  labels[i] << std::endl;
  //    std::cout <<  outputs_[i] << std::endl;
  //    }
}

}  // namespce of PaddleX
