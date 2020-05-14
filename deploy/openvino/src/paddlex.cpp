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
    executable_network_ = ie.LoadNetwork(network_, device);
    load_config(cfg_dir);
}

template <typename T>
void matU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) {
    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob) {
        THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
            << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto mblobHolder = mblob->wmap();

    T *blob_data = mblobHolder.as<T *>();

    cv::Mat resized_image(orig_image);
    if (static_cast<int>(width) != orig_image.size().width ||
            static_cast<int>(height) != orig_image.size().height) {
        cv::resize(orig_image, resized_image, cv::Size(width, height));
    }

    int batchOffset = batchIndex * width * height * channels;

    for (size_t c = 0; c < channels; c++) {
        for (size_t  h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[batchOffset + c * width * height + h * width + w] =
                        resized_image.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
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

bool Model::predict(cv::Mat* im, ClsResult* result) {
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
  InferRequest infer_request = executable_network_.CreateInferRequest();
  std::string input_name = network_.getInputsInfo().begin()->first;
  //im->convertTo(*im, CV_8UC3);
  Blob::Ptr input = infer_request.GetBlob(input_name);
  matU8ToBlob<float>(*im, input, 0);
  infer_request.Infer();
  std::string output_name = network_.getOutputsInfo().begin()->first;
  Blob::Ptr output = infer_request.GetBlob(output_name);
  MemoryBlob::CPtr moutput = as<MemoryBlob>(output);
  auto moutputHolder = moutput->rmap();
  outputs_ = moutputHolder.as<float *>();
  
  std::cout << sizeof(outputs_) << std::endl;
  // 对模型输出结果进行后处理
  auto ptr = std::max_element(outputs_, outputs_+sizeof(outputs_));
  result->category_id = std::distance(outputs_, ptr);
  result->score = *ptr;
  result->category = labels[result->category_id];
  //for (int i=0;i<sizeof(outputs_);i++){
  //    std::cout <<  labels[i] << std::endl; 
  //    std::cout <<  outputs_[i] << std::endl;
  //    }
}

}  // namespce of PaddleX
