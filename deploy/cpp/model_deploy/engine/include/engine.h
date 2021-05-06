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

#include "model_deploy/common/include/output_struct.h"

namespace PaddleDeploy {

struct InferenceConfig {
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

class InferEngine {
 public:
  virtual bool Init(const std::string &model_filename,
                    const std::string &params_filename,
                    const InferenceConfig &engine_config) = 0;

  virtual bool Infer(const std::vector<DataBlob> &inputs,
                     std::vector<DataBlob> *outputs) = 0;
};

}  // namespace PaddleDeploy
