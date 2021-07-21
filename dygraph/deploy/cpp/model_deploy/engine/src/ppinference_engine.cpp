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

#include "model_deploy/engine/include/ppinference_engine.h"

namespace PaddleDeploy {
bool Model::PaddleEngineInit(const PaddleEngineConfig& engine_config) {
  infer_engine_ = std::make_shared<PaddleInferenceEngine>();
  InferenceConfig config("paddle");
  *(config.paddle_config) = engine_config;
  return infer_engine_->Init(config);
}

bool PaddleInferenceEngine::Init(const InferenceConfig& infer_config) {
  const PaddleEngineConfig& engine_config = *(infer_config.paddle_config);
  paddle_infer::Config config;
  if ("" == engine_config.key) {
    config.SetModel(engine_config.model_filename,
                  engine_config.params_filename);
  } else {
#ifdef PADDLEX_DEPLOY_ENCRYPTION
    std::string model = decrypt_file(engine_config.model_filename.c_str(),
                                     engine_config.key.c_str());
    std::string params = decrypt_file(engine_config.params_filename.c_str(),
                                      engine_config.key.c_str());
    config.SetModelBuffer(model.c_str(),
                          model.size(),
                          params.c_str(),
                          params.size());
#else
    std::cerr << "Don't open with_encryption on compile" << std::endl;
    return false;
#endif  // PADDLEX_DEPLOY_ENCRYPTION
  }
  if (engine_config.use_mkl && !engine_config.use_gpu) {
    config.EnableMKLDNN();
    config.SetCpuMathLibraryNumThreads(engine_config.mkl_thread_num);
    config.SetMkldnnCacheCapacity(10);
  }
  if (engine_config.use_gpu) {
    config.EnableUseGpu(100, engine_config.gpu_id);
  } else {
    config.DisableGpu();
  }
  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  config.SwitchIrOptim(engine_config.use_ir_optim);
  config.EnableMemoryOptim();
  if (engine_config.use_trt && engine_config.use_gpu) {
    paddle_infer::PrecisionType precision;
    if (engine_config.precision == 0) {
      precision = paddle_infer::PrecisionType::kFloat32;
    } else if (engine_config.precision == 1) {
      precision = paddle_infer::PrecisionType::kHalf;
    } else if (engine_config.precision == 2) {
      precision = paddle_infer::PrecisionType::kInt8;
    } else {
      std::cerr << "Can not support the set precision" << std::endl;
      return false;
    }
    config.EnableTensorRtEngine(
        engine_config.max_workspace_size /* workspace_size*/,
        engine_config.max_batch_size /* max_batch_size*/,
        engine_config.min_subgraph_size /* min_subgraph_size*/,
        precision /* precision*/,
        engine_config.use_static /* use_static*/,
        engine_config.use_calib_mode /* use_calib_mode*/);

    if (engine_config.min_input_shape.size() != 0) {
      config.SetTRTDynamicShapeInfo(engine_config.min_input_shape,
                                    engine_config.max_input_shape,
                                    engine_config.optim_input_shape);
    }
  }
  predictor_ = std::move(paddle_infer::CreatePredictor(config));
  return true;
}

bool PaddleInferenceEngine::Infer(const std::vector<DataBlob>& inputs,
                                  std::vector<DataBlob>* outputs) {
  if (inputs.size() == 0) {
    std::cerr << "empty input image on PaddleInferenceEngine" << std::endl;
    return true;
  }
  auto input_names = predictor_->GetInputNames();
  for (int i = 0; i < inputs.size(); i++) {
    auto in_tensor = predictor_->GetInputHandle(input_names[i]);
    in_tensor->Reshape(inputs[i].shape);
    if (inputs[i].dtype == FLOAT32) {
      float* im_tensor_data;
      im_tensor_data = (float*)(inputs[i].data.data());  // NOLINT
      in_tensor->CopyFromCpu(im_tensor_data);
    } else if (inputs[i].dtype == INT64) {
      int64_t* im_tensor_data;
      im_tensor_data = (int64_t*)(inputs[i].data.data());  // NOLINT
      in_tensor->CopyFromCpu(im_tensor_data);
    } else if (inputs[i].dtype == INT32) {
      int* im_tensor_data;
      im_tensor_data = (int*)(inputs[i].data.data());  // NOLINT
      in_tensor->CopyFromCpu(im_tensor_data);
    } else if (inputs[i].dtype == INT8) {
      uint8_t* im_tensor_data;
      im_tensor_data = (uint8_t*)(inputs[i].data.data());  // NOLINT
      in_tensor->CopyFromCpu(im_tensor_data);
    } else {
      std::cerr << "There's unexpected input dtype: " << inputs[i].dtype
                << std::endl;
      return false;
    }
  }
  // predict
  predictor_->Run();

  // get output
  auto output_names = predictor_->GetOutputNames();
  for (const auto output_name : output_names) {
    auto output_tensor = predictor_->GetOutputHandle(output_name);
    auto output_tensor_shape = output_tensor->shape();
    DataBlob output;
    output.name = output_name;
    output.shape.assign(output_tensor_shape.begin(), output_tensor_shape.end());
    output.dtype = paddle_infer::DataType(output_tensor->type());
    output.lod = output_tensor->lod();
    int size = 1;
    for (const auto &i : output_tensor_shape) {
      size *= i;
    }
    if (output.dtype == 0) {
      output.data.resize(size * sizeof(float));
      output_tensor->CopyToCpu(reinterpret_cast<float *>(output.data.data()));
    } else if (output.dtype == 1) {
      output.data.resize(size * sizeof(int64_t));
      output_tensor->CopyToCpu(reinterpret_cast<int64_t *>(output.data.data()));
    } else if (output.dtype == 2) {
      output.data.resize(size * sizeof(int));
      output_tensor->CopyToCpu(reinterpret_cast<int *>(output.data.data()));
    } else if (output.dtype == 3) {
      output.data.resize(size * sizeof(uint8_t));
      output_tensor->CopyToCpu(reinterpret_cast<uint8_t *>(output.data.data()));
    }
    outputs->push_back(std::move(output));
  }
  return true;
}

}  //  namespace PaddleDeploy
