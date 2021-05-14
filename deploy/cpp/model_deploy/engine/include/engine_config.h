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

#include <string>
#include <vector>
#include <map>

namespace PaddleDeploy {

struct PaddleEngineConfig {
  //  model file path
  std::string model_filename = "";

  //  model params file path
  std::string params_filename = "";

  //  Whether to use mkdnn accelerator library when deploying on CPU
  bool use_mkl = true;

  //  The number of threads set when using mkldnn accelerator
  int mkl_thread_num = 8;

  //  Whether to use GPU
  bool use_gpu = false;

  //  Set GPU ID, default is 0
  int gpu_id = 0;

  //  Enable IR optimization
  bool use_ir_optim = true;

  // Whether to use TensorRT
  bool use_trt = false;

  //  Set batchsize
  int batch_size = 1;

  //  Set TensorRT min_subgraph_size
  int min_subgraph_size = 1;

  /*Set TensorRT data precision
  0: FP32
  1: FP16
  2: Int8
  */
  int precision = 0;

  //  When tensorrt is used, whether to serialize tensorrt engine to disk
  bool use_static = false;

  //  Is offline calibration required, when tensorrt is used
  bool use_calib_mode = false;
};

struct TritonEngineConfig {
  TritonEngineConfig() : model_name_(""), model_version_(""),
        request_id_(""), sequence_id_(0), sequence_start_(false),
        sequence_end_(false), priority_(0), server_timeout_(0),
        client_timeout_(0) {}
  /// The name of the model to run inference.
  std::string model_name_;
  /// The version of the model to use while running inference. The default
  /// value is an empty string which means the server will select the
  /// version of the model based on its internal policy.
  std::string model_version_;
  /// An identifier for the request. If specified will be returned
  /// in the response. Default value is an empty string which means no
  /// request_id will be used.
  std::string request_id_;
  /// The unique identifier for the sequence being represented by the
  /// object. Default value is 0 which means that the request does not
  /// belong to a sequence.
  uint64_t sequence_id_;
  /// Indicates whether the request being added marks the start of the
  /// sequence. Default value is False. This argument is ignored if
  /// 'sequence_id' is 0.
  bool sequence_start_;
  /// Indicates whether the request being added marks the end of the
  /// sequence. Default value is False. This argument is ignored if
  /// 'sequence_id' is 0.
  bool sequence_end_;
  /// Indicates the priority of the request. Priority value zero
  /// indicates that the default priority level should be used
  /// (i.e. same behavior as not specifying the priority parameter).
  /// Lower value priorities indicate higher priority levels. Thus
  /// the highest priority level is indicated by setting the parameter
  /// to 1, the next highest is 2, etc. If not provided, the server
  /// will handle the request using default setting for the model.
  uint64_t priority_;
  /// The timeout value for the request, in microseconds. If the request
  /// cannot be completed within the time by the server can take a
  /// model-specific action such as terminating the request. If not
  /// provided, the server will handle the request using default setting
  /// for the model.
  uint64_t server_timeout_;
  // The maximum end-to-end time, in microseconds, the request is allowed
  // to take. Note the HTTP library only offer the precision upto
  // milliseconds. The client will abort request when the specified time
  // elapses. The request will return error with message "Deadline Exceeded".
  // The default value is 0 which means client will wait for the
  // response from the server. This option is not supported for streaming
  // requests. Instead see 'stream_timeout' argument in
  // InferenceServerGrpcClient::StartStream().
  uint64_t client_timeout_;

  bool verbose_ = false;

  std::string url_;
};

struct TensorRTEngineConfig {
  std::string model_dir_;
  int max_workspace_size_;
  int max_batch_size_;
  std::string trt_cache_file_ = "";

  std::map<std::string, std::vector<int>> min_input_shape_;
  std::map<std::string, std::vector<int>> max_input_shape_;
  std::map<std::string, std::vector<int>> optim_input_shape_;

  void SetTRTDynamicShapeInfo(std::string input_name,
                            std::vector<int> min_input_shape,
                            std::vector<int> max_input_shape,
                            std::vector<int> optim_input_shape) {
    min_input_shape_[input_name] = min_input_shape;
    max_input_shape_[input_name] = max_input_shape;
    optim_input_shape_[input_name] = optim_input_shape;
  }
};

struct InferenceConfig {
  std::string engine_type;
  union {
    PaddleEngineConfig* paddle_config;
    TritonEngineConfig* triton_config;
    TensorRTEngineConfig* tensorrt_config;
  };

  InferenceConfig() {
    paddle_config = nullptr;
  }

  explicit InferenceConfig(std::string engine_type) {
    engine_type = engine_type;
    if ("paddle" == engine_type) {
      paddle_config = new PaddleEngineConfig();
    } else if ("triton" == engine_type) {
      triton_config = new TritonEngineConfig();
    } else if ("tensorrt" == engine_type) {
      tensorrt_config = new TensorRTEngineConfig();
    }
  }

  InferenceConfig(const InferenceConfig& config) {
    engine_type = config.engine_type;
    if ("paddle" == engine_type) {
      paddle_config = new PaddleEngineConfig();
      *paddle_config = *(config.paddle_config);
    } else if ("triton" == engine_type) {
      triton_config = new TritonEngineConfig();
      *triton_config = *(config.triton_config);
    } else if ("tensorrt" == engine_type) {
      tensorrt_config = new TensorRTEngineConfig();
      *tensorrt_config = *(config.tensorrt_config);
    }
  }

  ~InferenceConfig() {
    if ("paddle" == engine_type) {
      delete paddle_config;
      paddle_config = NULL;
    } else if ("triton" == engine_type) {
      delete triton_config;
      triton_config = NULL;
    } else if ("tensorrt" == engine_type) {
      delete tensorrt_config;
      tensorrt_config = NULL;
    }
  }
};

}  // namespace PaddleDeploy
