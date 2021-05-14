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
bool Model::TensorRTInit(const std::string& model_dir,
                         std::string trt_cache_file,
                         int max_workspace_size,
                         int max_batch_size) {
  infer_engine_ = std::make_shared<TensorRTInferenceEngine>();
  InferenceConfig config("tensorrt");
  config.tensorrt_config->model_dir_ = model_dir;
  config.tensorrt_config->max_workspace_size_ = max_workspace_size;
  config.tensorrt_config->max_batch_size_ = max_batch_size;
  config.tensorrt_config->trt_cache_file_ = trt_cache_file;
  // init input ?
  // config.tensorrt_config->SetTRTDynamicShapeInfo("inputs",
  //                                                 {1, 3, 224, 224},
  //                                                 {1, 3, 224, 224},
  //                                                 {1, 3, 224, 224});
  infer_engine_->Init(config);
}

bool TensorRTInferenceEngine::Init(const InferenceConfig& engine_config) {
  const TensorRTEngineConfig& tensorrt_config = *engine_config.tensorrt_config;
  std::ifstream engine_file(tensorrt_config.trt_cache_file_, std::ios::binary);
  if (engine_file) {
    std::cout << "load cached optimized tensorrt file." << std::endl;
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        LoadEngine(tensorrt_config.trt_cache_file_, logger_), InferDeleter());
    // nvinfer1::ICudaEngine* engine = LoadEngine(trt_cache_file_,
    // logger_);
    return false;
  }

  auto builder = InferUniquePtr<nvinfer1::IBuilder>(
                     nvinfer1::createInferBuilder(logger_));
  auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(
                     builder->createBuilderConfig());
  auto profile = builder->createOptimizationProfile();

  for (auto input_shape : tensorrt_config.min_input_shape_) {
    nvinfer1::Dims input_dims;
    input_dims.nbDims = input_shape.second.size();
    for (int i = 0; i < input_shape.second.size(); i++) {
      input_dims.d[i] = input_shape.second[i];
    }
    profile->setDimensions(input_shape.first.c_str(),
                           nvinfer1::OptProfileSelector::kMIN, input_dims);
  }
  for (auto input_shape : tensorrt_config.max_input_shape_) {
    nvinfer1::Dims input_dims;
    input_dims.nbDims = input_shape.second.size();
    for (int i = 0; i < input_shape.second.size(); i++) {
      input_dims.d[i] = input_shape.second[i];
    }
    profile->setDimensions(input_shape.first.c_str(),
                           nvinfer1::OptProfileSelector::kMAX, input_dims);
  }
  for (auto input_shape : tensorrt_config.optim_input_shape_) {
    nvinfer1::Dims input_dims;
    input_dims.nbDims = input_shape.second.size();
    for (int i = 0; i < input_shape.second.size(); i++) {
      input_dims.d[i] = input_shape.second[i];
    }
    profile->setDimensions(input_shape.first.c_str(),
                           nvinfer1::OptProfileSelector::kOPT, input_dims);
  }

  config->addOptimizationProfile(profile);

  config->setMaxWorkspaceSize(tensorrt_config.max_workspace_size_);

  const auto explicitBatch =
      tensorrt_config.max_batch_size_ << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(explicitBatch));

  auto parser = InferUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger_));
  auto parsed = parser->parseFromFile(
      tensorrt_config.model_dir_.c_str(),
      static_cast<int>(logger_.mReportableSeverity));

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      builder->buildEngineWithConfig(*network, *config), InferDeleter());
  SaveEngine(*(engine_.get()), tensorrt_config.trt_cache_file_);
  return true;
}

void TensorRTInferenceEngine::FeedInput(
    const std::vector<DataBlob>& input_blobs,
    const TensorRT::BufferManager& buffers) {
  for (auto input_blob : input_blobs) {
    if (input_blob.dtype == 0) {
      float *hostDataBuffer =
          reinterpret_cast<float*>(buffers.getHostBuffer(input_blob.name));
      hostDataBuffer =  reinterpret_cast<float*>(input_blob.data.data());
    } else if (input_blob.dtype == 1) {
      int64_t *hostDataBuffer =
          reinterpret_cast<int64_t*>(buffers.getHostBuffer(input_blob.name));
      hostDataBuffer = reinterpret_cast<int64_t*>(input_blob.data.data());
    } else if (input_blob.dtype == 2) {
      int *hostDataBuffer =
          reinterpret_cast<int*>(buffers.getHostBuffer(input_blob.name));
      hostDataBuffer =  reinterpret_cast<int*>(input_blob.data.data());
    } else if (input_blob.dtype == 3) {
      uint8_t *hostDataBuffer =
          reinterpret_cast<uint8_t*>(buffers.getHostBuffer(input_blob.name));
      hostDataBuffer =  reinterpret_cast<uint8_t*>(input_blob.data.data());
    }
  }
}

nvinfer1::ICudaEngine* TensorRTInferenceEngine::LoadEngine(
                                            const std::string& engine,
                                            NaiveLogger logger, 
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
      nvinfer1::createInferRuntime(logger)};

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
  auto context = InferUniquePtr<nvinfer1::IExecutionContext>(
                     engine_->createExecutionContext());
  int input_index = 0;
  for (auto input_blob : input_blobs) {
    nvinfer1::Dims input_dims;
    input_dims.nbDims = input_blob.shape.size();
    for (int i = 0; i < input_blob.shape.size(); i++) {
      input_dims.d[i] = input_blob.shape[i];
    }
    context->setBindingDimensions(input_index, input_dims);
    input_index++;
  }
  // const int batch_size = 0;
  TensorRT::BufferManager buffers(engine_, 0, context.get());
  FeedInput(input_blobs, buffers);
  buffers.copyInputToDevice();
  bool status = context->executeV2(buffers.getDeviceBindings().data());
  buffers.copyOutputToHostAsync();
  return status;
}

}  //  namespace PaddleDeploy
