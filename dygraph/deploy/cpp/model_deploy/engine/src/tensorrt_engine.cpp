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

#include "model_deploy/engine/include/tensorrt_engine.h"

namespace PaddleDeploy {
int DtypeConver(const nvinfer1::DataType& dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kINT32:
      return 2;
    case nvinfer1::DataType::kFLOAT:
      return 0;
    case nvinfer1::DataType::kBOOL:
      return 3;
    case nvinfer1::DataType::kINT8:
      return 3;
  }
  std::cerr << "Fail trt dtype";
  return -1;
}

bool Model::TensorRTInit(const TensorRTEngineConfig& engine_config) {
  infer_engine_ = std::make_shared<TensorRTInferenceEngine>();
  InferenceConfig config("tensorrt");

  YAML::Node node  = YAML::LoadFile(engine_config.cfg_file_);
  if (!node["input"].IsDefined()) {
    std::cout << "Fail to find input in yaml file!" << std::endl;
    return false;
  }
  if (!node["output"].IsDefined()) {
    std::cout << "Fail to find output in yaml file!" << std::endl;
    return false;
  }

  *(config.tensorrt_config) = engine_config;
  config.tensorrt_config->yaml_config_ = node;
  return infer_engine_->Init(config);
}

bool TensorRTInferenceEngine::Init(const InferenceConfig& engine_config) {
  const TensorRTEngineConfig& tensorrt_config = *engine_config.tensorrt_config;

  TensorRT::setCudaDevice(tensorrt_config.gpu_id_);

  std::ifstream engine_file(tensorrt_config.trt_cache_file_, std::ios::binary);
  if (engine_file) {
    std::cout << "Start load cached optimized tensorrt file." << std::endl;
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
                             LoadEngine(tensorrt_config.trt_cache_file_),
                             InferDeleter());
    if (!engine_) {
      std::cerr << "Fail load cached optimized tensorrt" << std::endl;
      return false;
    }
    return true;
  }

  auto builder = InferUniquePtr<nvinfer1::IBuilder>(
                     nvinfer1::createInferBuilder(logger_));
  if (!builder) {
    return false;
  }

  const auto explicitBatch = 1U << static_cast<uint32_t>(
                 nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(
                     builder->createNetworkV2(explicitBatch));
  if (!network) {
    return false;
  }

  auto parser = InferUniquePtr<nvonnxparser::IParser>(
                    nvonnxparser::createParser(*network, logger_));
  if (!parser) {
    return false;
  }
  if (!parser->parseFromFile(tensorrt_config.model_file_.c_str(),
                             static_cast<int>(logger_.mReportableSeverity))) {
    return false;
  }

  auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(
                     builder->createBuilderConfig());
  if (!config) {
    return false;
  }

  config->setMaxWorkspaceSize(tensorrt_config.max_workspace_size_);

  // set shape.  Currently don't support dynamic shapes
  yaml_config_ = tensorrt_config.yaml_config_["output"];
  auto profile = builder->createOptimizationProfile();
  for (const auto& input : tensorrt_config.yaml_config_["input"]) {
    nvinfer1::Dims input_dims;
    input_dims.nbDims = static_cast<int>(input["dims"].size());
    for (int i = 0; i < input_dims.nbDims; ++i) {
      input_dims.d[i] = input["dims"][i].as<int>();
      if (input_dims.d[i] < 0) {
        std::cerr << "Fail input shape on yaml file" << std::endl;
        return false;
      }
    }
    profile->setDimensions(input["name"].as<std::string>().c_str(),
                          nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input["name"].as<std::string>().c_str(),
                          nvinfer1::OptProfileSelector::kMAX, input_dims);
    profile->setDimensions(input["name"].as<std::string>().c_str(),
                          nvinfer1::OptProfileSelector::kOPT, input_dims);
  }
  config->addOptimizationProfile(profile);

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
                    builder->buildEngineWithConfig(*network,
                                                   *config),
                    InferDeleter());

  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
                    engine_->createExecutionContext(),
                    InferDeleter());
  if (!context) {
    return false;
  }

  if (tensorrt_config.save_engine_) {
    if (!SaveEngine(*(engine_.get()), tensorrt_config.trt_cache_file_)) {
      std::cout << "Fail save Trt Engine to "
                << tensorrt_config.trt_cache_file_ << std::endl;
    }
  }
  return true;
}

void TensorRTInferenceEngine::FeedInput(
         const std::vector<DataBlob>& input_blobs,
         const TensorRT::BufferManager& buffers) {
  for (auto input_blob : input_blobs) {
    int size = std::accumulate(input_blob.shape.begin(),
                    input_blob.shape.end(), 1, std::multiplies<int>());
    if (input_blob.dtype == 0) {
      float* hostDataBuffer =
          reinterpret_cast<float*>(buffers.getHostBuffer(input_blob.name));
      memcpy(hostDataBuffer,
             reinterpret_cast<float*>(input_blob.data.data()),
             size * sizeof(float));
    } else if (input_blob.dtype == 1) {
      int64_t* hostDataBuffer =
          reinterpret_cast<int64_t*>(buffers.getHostBuffer(input_blob.name));
      memcpy(hostDataBuffer,
             reinterpret_cast<int64_t*>(input_blob.data.data()),
             size * sizeof(int64_t));
    } else if (input_blob.dtype == 2) {
      int* hostDataBuffer =
          reinterpret_cast<int*>(buffers.getHostBuffer(input_blob.name));
      memcpy(hostDataBuffer,
             reinterpret_cast<int*>(input_blob.data.data()),
             size * sizeof(int));
    } else if (input_blob.dtype == 3) {
      uint8_t* hostDataBuffer =
          reinterpret_cast<uint8_t*>(buffers.getHostBuffer(input_blob.name));
      memcpy(hostDataBuffer,
             reinterpret_cast<uint8_t*>(input_blob.data.data()),
             size * sizeof(uint8_t));
    }
  }
}

nvinfer1::ICudaEngine* TensorRTInferenceEngine::LoadEngine(
                                            const std::string& engine,
                                            int DLACore) {
  std::ifstream engine_file(engine, std::ios::binary);
  if (!engine_file) {
    std::cerr << "Error opening engine file: " << engine << std::endl;
    return nullptr;
  }

  engine_file.seekg(0, engine_file.end);
  int64_t fsize = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);

  std::vector<char> engineData(fsize);
  engine_file.read(engineData.data(), fsize);
  if (!engine_file) {
    std::cerr << "Error loading engine file: " << engine << std::endl;
    return nullptr;
  }

  InferUniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(logger_)};

  if (DLACore != -1) {
    runtime->setDLACore(DLACore);
  }
  return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
}

bool TensorRTInferenceEngine::SaveEngine(const nvinfer1::ICudaEngine& engine,
                                         const std::string& file_name) {
  std::ofstream engine_file(file_name, std::ios::binary);
  if (!engine_file) {
    return false;
  }

  InferUniquePtr<nvinfer1::IHostMemory> serializedEngine{engine.serialize()};
  if (serializedEngine == nullptr) {
    return false;
  }

  engine_file.write(reinterpret_cast<char *>(serializedEngine->data()),
                    serializedEngine->size());
  return !engine_file.fail();
}

bool TensorRTInferenceEngine::Infer(const std::vector<DataBlob>& input_blobs,
                                    std::vector<DataBlob>* output_blobs) {
  TensorRT::BufferManager buffers(engine_);
  FeedInput(input_blobs, buffers);
  buffers.copyInputToDevice();
  bool status = context->executeV2(buffers.getDeviceBindings().data());
  if (!status) {
    return false;
  }
  buffers.copyOutputToHost();

  for (const auto& output_config : yaml_config_) {
    std::string output_name = output_config["name"].as<std::string>();
    int index = engine_->getBindingIndex(output_name.c_str());
    nvinfer1::DataType dtype = engine_->getBindingDataType(index);

    DataBlob output_blob;
    output_blob.name = output_name;
    output_blob.dtype = DtypeConver(dtype);
    for (auto shape : output_config["dims"]) {
      output_blob.shape.push_back(shape.as<int>());
    }

    size_t size = std::accumulate(output_blob.shape.begin(),
                    output_blob.shape.end(), 1, std::multiplies<size_t>());
    if (output_blob.dtype == 0) {
      assert(size * sizeof(float) == buffers.size(output_name));
      float* output = static_cast<float*>(buffers.getHostBuffer(output_name));
      output_blob.data.resize(size * sizeof(float));
      memcpy(output_blob.data.data(), output, size * sizeof(float));
    } else if (output_blob.dtype == 1) {
      assert(size * sizeof(int64_t) == buffers.size(output_name));
      int64_t* output = static_cast<int64_t*>(
                            buffers.getHostBuffer(output_name));
      output_blob.data.resize(size * sizeof(int64_t));
      memcpy(output_blob.data.data(), output, size * sizeof(int64_t));
    } else if (output_blob.dtype == 2) {
      assert(size * sizeof(int) == buffers.size(output_name));
      int* output = static_cast<int*>(buffers.getHostBuffer(output_name));
      output_blob.data.resize(size * sizeof(int));
      memcpy(output_blob.data.data(), output, size * sizeof(int));
    } else if (output_blob.dtype == 3) {
      assert(size * sizeof(uint8_t) == buffers.size(output_name));
      uint8_t* output = static_cast<uint8_t*>(
                            buffers.getHostBuffer(output_name));
      output_blob.data.resize(size * sizeof(uint8_t));
      memcpy(output_blob.data.data(), output, size * sizeof(uint8_t));
    }

    output_blobs->push_back(std::move(output_blob));
  }
  return true;
}

}  //  namespace PaddleDeploy
