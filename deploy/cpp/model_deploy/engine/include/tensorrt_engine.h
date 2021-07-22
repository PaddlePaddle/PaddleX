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

#pragma once

#include <iostream>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"

#include "model_deploy/common/include/output_struct.h"
#include "model_deploy/engine/include/engine.h"
#include "model_deploy/common/include/base_model.h"
#include "model_deploy/engine/include/tensorrt_buffers.h"

namespace PaddleDeploy {

using Severity = nvinfer1::ILogger::Severity;

struct InferDeleter {
  template <typename T> void operator()(T *obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

// A logger for create TensorRT infer builder.
class NaiveLogger : public nvinfer1::ILogger {
 public:
  explicit NaiveLogger(Severity severity = Severity::kWARNING)
      : mReportableSeverity(severity) {}

  void log(nvinfer1::ILogger::Severity severity, const char *msg) override {
    switch (severity) {
    case Severity::kINFO:
      LOG(INFO) << msg;
      break;
    case Severity::kWARNING:
      LOG(WARNING) << msg;
      break;
    case Severity::kINTERNAL_ERROR:
      std::cout << "kINTERNAL_ERROR:" << msg << std::endl;
      break;
    case Severity::kERROR:
      LOG(ERROR) << msg;
      break;
    case Severity::kVERBOSE:
      // std::cout << "kVERBOSE:" << msg << std::endl;
      break;
    default:
      // std::cout << "default:" << msg << std::endl;
      break;
    }
  }

  static NaiveLogger &Global() {
    static NaiveLogger *x = new NaiveLogger;
    return *x;
  }

  ~NaiveLogger() override {}

  Severity mReportableSeverity;
};

class TensorRTInferenceEngine : public InferEngine {
  template <typename T>
  using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

 public:
  bool Init(const InferenceConfig& engine_config);

  bool Infer(const std::vector<DataBlob>& input_blobs,
             std::vector<DataBlob>* output_blobs);

  std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  NaiveLogger logger_;

 private:
  void FeedInput(const std::vector<DataBlob>& input_blobs,
                 const TensorRT::BufferManager& buffers);

  bool SaveEngine(const nvinfer1::ICudaEngine& engine,
                  const std::string& fileName);

  nvinfer1::ICudaEngine* LoadEngine(const std::string& engine,
                                    int DLACore = -1);

  void ParseONNXModel(const std::string& model_dir);

  YAML::Node yaml_config_;
};

}  //  namespace PaddleDeploy
