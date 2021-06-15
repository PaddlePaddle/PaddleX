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

#include "model_deploy/engine/include/openvino_engine.h"

namespace PaddleDeploy {
bool Model::OpenVinoEngineInit(const OpenVinoEngineConfig& engine_config) {
  infer_engine_ = std::make_shared<OpenVinoEngine>();
  InferenceConfig config("openvino");
  *(config.openvino_config) = engine_config;
  return infer_engine_->Init(config);
}

bool OpenVinoEngine::Init(const InferenceConfig& infer_config) {
  const OpenVinoEngineConfig& engine_config = *(infer_config.openvino_config);
  InferenceEngine::Core ie;
  network_ = ie.ReadNetwork(engine_config.xml_file_, engine_config.bin_file_);
  network_.setBatchSize(engine_config.batch_size_);
  if (engine_config.device_ == "MYRIAD") {
    std::map<std::string, std::string> networkConfig;
    networkConfig["VPU_HW_STAGES_OPTIMIZATION"] = "NO";
    executable_network_ = ie.LoadNetwork(
            network_, engine_config.device_, networkConfig);
  } else {
    executable_network_ = ie.LoadNetwork(network_, engine_config.device_);
  }
  return true;
}

bool OpenVinoEngine::Infer(const std::vector<DataBlob> &inputs,
                          std::vector<DataBlob> *outputs) {
  InferenceEngine::InferRequest infer_request =
        executable_network_.CreateInferRequest();
  for (int i = 0; i < inputs.size(); i++) {
    InferenceEngine::TensorDesc input_tensor;
    InferenceEngine::Blob::Ptr input_blob =
        infer_request.GetBlob(inputs[i].name);
    InferenceEngine::MemoryBlob::Ptr input_mem_blob =
        InferenceEngine::as<InferenceEngine::MemoryBlob>(input_blob);
    auto mem_blob_holder = input_mem_blob->wmap();
    int size = std::accumulate(inputs[i].shape.begin(),
                    inputs[i].shape.end(), 1, std::multiplies<int>());
    if (inputs[i].dtype == 0) {
      input_tensor.setPrecision(InferenceEngine::Precision::FP32);
      float *blob_data = mem_blob_holder.as<float *>();
      memcpy(blob_data, inputs[i].data.data(), size * sizeof(float));
    } else if (inputs[i].dtype == 1) {
      input_tensor.setPrecision(InferenceEngine::Precision::U64);
      int64_t *blob_data = mem_blob_holder.as<int64_t *>();
      memcpy(blob_data, inputs[i].data.data(), size * sizeof(int64_t));
    } else if (inputs[i].dtype == 2) {
      input_tensor.setPrecision(InferenceEngine::Precision::I32);
      int *blob_data = mem_blob_holder.as<int *>();
      memcpy(blob_data, inputs[i].data.data(), size * sizeof(int));
    } else if (inputs[i].dtype == 3) {
      input_tensor.setPrecision(InferenceEngine::Precision::U8);
      uint8_t *blob_data = mem_blob_holder.as<uint8_t *>();
      memcpy(blob_data, inputs[i].data.data(), size * sizeof(uint8_t));
      infer_request.SetBlob(inputs[i].name, input_blob);
    }
  }

  // do inference
  infer_request.Infer();

  InferenceEngine::OutputsDataMap out_maps = network_.getOutputsInfo();
  for (const auto & output_map : out_maps) {
    DataBlob output;
    std::string name = output_map.first;
    output.name = name;
    InferenceEngine::Blob::Ptr output_ptr = infer_request.GetBlob(name);
    InferenceEngine::MemoryBlob::CPtr moutput =
      InferenceEngine::as<InferenceEngine::MemoryBlob>(output_ptr);
    InferenceEngine::TensorDesc blob_output = moutput->getTensorDesc();
    InferenceEngine::SizeVector output_shape = blob_output.getDims();
    int size = 1;
    output.shape.clear();
    for (auto& i : output_shape) {
      size *= i;
      output.shape.push_back(static_cast<int>(i));
    }
    GetDtype(blob_output, &output);
    auto moutputHolder = moutput->rmap();
    if (output.dtype == 0) {
      output.data.resize(size * sizeof(float));
      float* data = moutputHolder.as<float *>();
      memcpy(output.data.data(), data, size * sizeof(float));
    } else if (output.dtype == 1) {
      output.data.resize(size * sizeof(int64_t));
      int64_t* data = moutputHolder.as<int64_t *>();
      memcpy(output.data.data(), data, size * sizeof(int64_t));
    } else if (output.dtype == 2) {
      output.data.resize(size * sizeof(int));
      int* data = moutputHolder.as<int *>();
      memcpy(output.data.data(), data, size * sizeof(int));
    } else if (output.dtype == 3) {
      output.data.resize(size * sizeof(uint8_t));
      uint8_t* data = moutputHolder.as<uint8_t *>();
      memcpy(output.data.data(), data, size * sizeof(uint8_t));
    }
    outputs->push_back(std::move(output));
  }
  return true;
}

bool OpenVinoEngine::GetDtype(const InferenceEngine::TensorDesc &output_blob,
                          DataBlob *output) {
  InferenceEngine::Precision output_precision = output_blob.getPrecision();
  if (output_precision == 10) {
    output->dtype = 0;
  } else if (output_precision == 73) {
    output->dtype = 1;
  } else if (output_precision == 70) {
    output->dtype = 2;
  } else if (output_precision == 40) {
    output->dtype = 3;
  } else {
    std::cout << "can't paser the precision type" << std::endl;
    return false;
  }
  return true;
}

}  //  namespace PaddleDeploy
