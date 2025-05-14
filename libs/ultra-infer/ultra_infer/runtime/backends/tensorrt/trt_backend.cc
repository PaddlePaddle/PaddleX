// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "ultra_infer/runtime/backends/tensorrt/trt_backend.h"

#include <cstring>
#include <unordered_map>

#include "NvInferRuntime.h"
#include "ultra_infer/function/cuda_cast.h"
#include "ultra_infer/utils/utils.h"
#ifdef ENABLE_PADDLE2ONNX
#include "paddle2onnx/converter.h"
#endif

namespace ultra_infer {

FDTrtLogger *FDTrtLogger::logger = nullptr;

// Check if the model can build tensorrt engine now
// If the model has dynamic input shape, it will require defined shape
// information We can set the shape range information by function
// SetTrtInputShape() But if the shape range is not defined, then the engine
// cannot build, in this case, The engine will build once there's data fed,
// and the shape range will be updated
bool CanBuildEngine(
    const std::map<std::string, ShapeRangeInfo> &shape_range_info) {
  for (auto iter = shape_range_info.begin(); iter != shape_range_info.end();
       ++iter) {
    bool is_full_static = true;
    for (size_t i = 0; i < iter->second.shape.size(); ++i) {
      if (iter->second.shape[i] < 0) {
        is_full_static = false;
        break;
      }
    }

    if (is_full_static) {
      continue;
    }
    for (size_t i = 0; i < iter->second.shape.size(); ++i) {
      if (iter->second.min[i] < 0 || iter->second.max[i] < 0) {
        return false;
      }
    }
  }
  return true;
}

bool TrtBackend::LoadTrtCache(const std::string &trt_engine_file) {
  cudaSetDevice(option_.gpu_id);

  std::string engine_buffer;
  if (!ReadBinaryFromFile(trt_engine_file, &engine_buffer)) {
    FDERROR << "Failed to load TensorRT Engine from " << trt_engine_file << "."
            << std::endl;
    return false;
  }

  FDUniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(*FDTrtLogger::Get())};
  if (!runtime) {
    FDERROR << "Failed to call createInferRuntime()." << std::endl;
    return false;
  }
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engine_buffer.data(),
                                     engine_buffer.size()),
      FDInferDeleter());
  if (!engine_) {
    FDERROR << "Failed to call deserializeCudaEngine()." << std::endl;
    return false;
  }

  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  GetInputOutputInfo();

  for (int32_t i = 0; i < engine_->getNbBindings(); ++i) {
    if (!engine_->bindingIsInput(i)) {
      continue;
    }
    auto min = ToVec(engine_->getProfileDimensions(
        i, 0, nvinfer1::OptProfileSelector::kMAX));
    auto max = ToVec(engine_->getProfileDimensions(
        i, 0, nvinfer1::OptProfileSelector::kMIN));
    auto name = std::string(engine_->getBindingName(i));
    auto iter = shape_range_info_.find(name);
    if (iter == shape_range_info_.end()) {
      FDERROR << "There's no input named '" << name << "' in loaded model."
              << std::endl;
      return false;
    }
    iter->second.Update(min);
    iter->second.Update(max);
  }
  FDINFO << "Build TensorRT Engine from cache file: " << trt_engine_file
         << " with shape range information as below," << std::endl;
  for (const auto &item : shape_range_info_) {
    FDINFO << item.second << std::endl;
  }
  return true;
}

bool TrtBackend::Init(const RuntimeOption &runtime_option) {
  auto trt_option = runtime_option.trt_option;
  trt_option.model_file = runtime_option.model_file;
  trt_option.params_file = runtime_option.params_file;
  trt_option.model_format = runtime_option.model_format;
  trt_option.gpu_id = runtime_option.device_id;
  trt_option.enable_pinned_memory = runtime_option.enable_pinned_memory;
  trt_option.external_stream_ = runtime_option.external_stream_;
  if (runtime_option.device != Device::GPU) {
    FDERROR << "TrtBackend only supports Device::GPU, but now it's "
            << runtime_option.device << "." << std::endl;
    return false;
  }
  if (runtime_option.model_format != ModelFormat::PADDLE &&
      runtime_option.model_format != ModelFormat::ONNX) {
    FDERROR
        << "TrtBackend only supports model format PADDLE/ONNX, but now it's "
        << runtime_option.model_format << "." << std::endl;
    return false;
  }
  if (runtime_option.model_format == ModelFormat::PADDLE) {
    if (runtime_option.model_from_memory_) {
      return InitFromPaddle(runtime_option.model_file,
                            runtime_option.params_file, trt_option);
    } else {
      std::string model_buffer;
      std::string params_buffer;
      FDASSERT(ReadBinaryFromFile(runtime_option.model_file, &model_buffer),
               "Failed to read model file %s.",
               runtime_option.model_file.c_str());
      FDASSERT(ReadBinaryFromFile(runtime_option.params_file, &params_buffer),
               "Failed to read parameters file %s.",
               runtime_option.params_file.c_str());
      return InitFromPaddle(model_buffer, params_buffer, trt_option);
    }
  } else {
    if (runtime_option.model_from_memory_) {
      return InitFromOnnx(runtime_option.model_file, trt_option);
    } else {
      std::string model_buffer;
      FDASSERT(ReadBinaryFromFile(runtime_option.model_file, &model_buffer),
               "Failed to read model file %s.",
               runtime_option.model_file.c_str());
      return InitFromOnnx(model_buffer, trt_option);
    }
  }
  return true;
}

bool TrtBackend::InitFromPaddle(const std::string &model_buffer,
                                const std::string &params_buffer,
                                const TrtBackendOption &option, bool verbose) {
  if (initialized_) {
    FDERROR << "TrtBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  option_ = option;

#ifdef ENABLE_PADDLE2ONNX
  std::vector<paddle2onnx::CustomOp> ops;
  ops.resize(1);
  strcpy(ops[0].op_name, "pool2d");
  strcpy(ops[0].export_op_name, "AdaptivePool2d");
  char *model_content_ptr;
  int model_content_size = 0;
  char *calibration_cache_ptr;
  int calibration_cache_size = 0;
  if (!paddle2onnx::Export(model_buffer.c_str(), model_buffer.size(),
                           params_buffer.c_str(), params_buffer.size(),
                           &model_content_ptr, &model_content_size, 11, true,
                           verbose, true, true, true, ops.data(), 1, "tensorrt",
                           &calibration_cache_ptr, &calibration_cache_size, "",
                           &save_external_)) {
    FDERROR << "Error occurred while export PaddlePaddle to ONNX format."
            << std::endl;
    return false;
  }
  std::string onnx_model_proto(model_content_ptr,
                               model_content_ptr + model_content_size);
  delete[] model_content_ptr;
  model_content_ptr = nullptr;
  if (calibration_cache_size) {
    std::string calibration_str(calibration_cache_ptr,
                                calibration_cache_ptr + calibration_cache_size);
    calibration_str_ = calibration_str;
    delete[] calibration_cache_ptr;
  }
  if (save_external_) {
    model_file_name_ = "model.onnx";
    std::fstream f(model_file_name_, std::ios::out);
    FDASSERT(f.is_open(), "Can not open file: %s to save model.",
             model_file_name_.c_str());
    f << onnx_model_proto;
    f.close();
  }
  return InitFromOnnx(onnx_model_proto, option);
#else
  FDERROR << "Didn't compile with PaddlePaddle frontend, you can try to "
             "call `InitFromOnnx` instead."
          << std::endl;
  return false;
#endif
}

bool TrtBackend::InitFromOnnx(const std::string &model_buffer,
                              const TrtBackendOption &option) {
  if (initialized_) {
    FDERROR << "TrtBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  option_ = option;
  cudaSetDevice(option_.gpu_id);

  std::string onnx_content = model_buffer;

  // This part of code will record the original outputs order
  // because the converted tensorrt network may exist wrong order of outputs
  outputs_order_.clear();
  auto onnx_reader =
      paddle2onnx::OnnxReader(onnx_content.c_str(), onnx_content.size());
  for (int i = 0; i < onnx_reader.num_outputs; ++i) {
    std::string name(onnx_reader.outputs[i].name);
    outputs_order_[name] = i;
  }

  shape_range_info_.clear();
  inputs_desc_.clear();
  outputs_desc_.clear();
  inputs_desc_.resize(onnx_reader.num_inputs);
  outputs_desc_.resize(onnx_reader.num_outputs);
  for (int i = 0; i < onnx_reader.num_inputs; ++i) {
    std::string name(onnx_reader.inputs[i].name);
    std::vector<int64_t> shape(onnx_reader.inputs[i].shape,
                               onnx_reader.inputs[i].shape +
                                   onnx_reader.inputs[i].rank);
    inputs_desc_[i].name = name;
    inputs_desc_[i].shape.assign(shape.begin(), shape.end());
    inputs_desc_[i].dtype = ReaderDtypeToTrtDtype(onnx_reader.inputs[i].dtype);
    inputs_desc_[i].original_dtype =
        ReaderDtypeToFDDtype(onnx_reader.inputs[i].dtype);
    auto info = ShapeRangeInfo(shape);
    info.name = name;
    auto iter_min = option.min_shape.find(name);
    auto iter_max = option.max_shape.find(name);
    auto iter_opt = option.opt_shape.find(name);
    if (iter_min != option.min_shape.end()) {
      info.min.assign(iter_min->second.begin(), iter_min->second.end());
      info.max.assign(iter_max->second.begin(), iter_max->second.end());
      info.opt.assign(iter_opt->second.begin(), iter_opt->second.end());
    }
    shape_range_info_.insert(std::make_pair(name, info));
  }

  for (int i = 0; i < onnx_reader.num_outputs; ++i) {
    std::string name(onnx_reader.outputs[i].name);
    std::vector<int64_t> shape(onnx_reader.outputs[i].shape,
                               onnx_reader.outputs[i].shape +
                                   onnx_reader.outputs[i].rank);
    outputs_desc_[i].name = name;
    outputs_desc_[i].shape.assign(shape.begin(), shape.end());
    outputs_desc_[i].dtype =
        ReaderDtypeToTrtDtype(onnx_reader.outputs[i].dtype);
    outputs_desc_[i].original_dtype =
        ReaderDtypeToFDDtype(onnx_reader.outputs[i].dtype);
  }

  if (option_.external_stream_) {
    stream_ = reinterpret_cast<cudaStream_t>(option_.external_stream_);
  } else {
    FDASSERT(cudaStreamCreate(&stream_) == 0,
             "[ERROR] Error occurs while calling cudaStreamCreate().");
  }

  if (save_external_) {
    onnx_content.clear();
    onnx_content = model_file_name_;
  }
  if (!CreateTrtEngineFromOnnx(onnx_content)) {
    FDERROR << "Failed to create tensorrt engine." << std::endl;
    return false;
  }
  initialized_ = true;
  return true;
}

int TrtBackend::ShapeRangeInfoUpdated(const std::vector<FDTensor> &inputs) {
  bool need_update_engine = false;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto iter = shape_range_info_.find(inputs[i].name);
    if (iter == shape_range_info_.end()) {
      FDERROR << "There's no input named '" << inputs[i].name
              << "' in loaded model." << std::endl;
    }
    if (iter->second.Update(inputs[i].shape) == 1) {
      need_update_engine = true;
    }
  }
  return need_update_engine;
}

bool TrtBackend::Infer(std::vector<FDTensor> &inputs,
                       std::vector<FDTensor> *outputs, bool copy_to_fd) {
  if (inputs.size() != NumInputs()) {
    FDERROR << "Require " << NumInputs() << "inputs, but get " << inputs.size()
            << "." << std::endl;
    return false;
  }
  if (ShapeRangeInfoUpdated(inputs)) {
    // meet new shape output of predefined max/min shape
    // rebuild the tensorrt engine
    FDWARNING
        << "TensorRT engine will be rebuilt once shape range information "
           "changed, this may take lots of time, you can set a proper shape "
           "range before loading model to avoid rebuilding process. refer "
           "https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/"
           "faq/"
           "tensorrt_tricks.md for more details."
        << std::endl;
    BuildTrtEngine();
  }

  RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN
  cudaSetDevice(option_.gpu_id);
  SetInputs(inputs);
  AllocateOutputsBuffer(outputs, copy_to_fd);

  RUNTIME_PROFILE_LOOP_BEGIN(1)
  if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
    FDERROR << "Failed to Infer with TensorRT." << std::endl;
    return false;
  }
  RUNTIME_PROFILE_LOOP_END

  for (size_t i = 0; i < outputs->size(); ++i) {
    // if the final output tensor's dtype is different from the model output
    // tensor's dtype, then we need cast the data to the final output's dtype
    auto model_output_dtype =
        GetFDDataType(outputs_device_buffer_[(*outputs)[i].name].dtype());
    if ((*outputs)[i].dtype != model_output_dtype) {
      FDTensor output_tensor;
      output_tensor.SetExternalData(
          (*outputs)[i].shape, model_output_dtype,
          outputs_device_buffer_[(*outputs)[i].name].data(), Device::GPU);

      casted_output_tensors_[(*outputs)[i].name].Resize(
          (*outputs)[i].shape, (*outputs)[i].dtype, (*outputs)[i].name,
          Device::GPU);
      function::CudaCast(output_tensor,
                         &casted_output_tensors_[(*outputs)[i].name], stream_);
      if (!copy_to_fd) {
        (*outputs)[i].SetExternalData(
            (*outputs)[i].shape, model_output_dtype,
            casted_output_tensors_[(*outputs)[i].name].MutableData(),
            Device::GPU, option_.gpu_id);
      }
    } else {
      casted_output_tensors_[(*outputs)[i].name].SetExternalData(
          (*outputs)[i].shape, model_output_dtype,
          outputs_device_buffer_[(*outputs)[i].name].data(), Device::GPU);
    }
  }
  if (copy_to_fd) {
    for (size_t i = 0; i < outputs->size(); ++i) {
      FDASSERT(
          cudaMemcpyAsync((*outputs)[i].Data(),
                          casted_output_tensors_[(*outputs)[i].name].Data(),
                          (*outputs)[i].Nbytes(), cudaMemcpyDeviceToHost,
                          stream_) == 0,
          "[ERROR] Error occurs while copy memory from GPU to CPU.");
    }
    FDASSERT(cudaStreamSynchronize(stream_) == cudaSuccess,
             "[ERROR] Error occurs while sync cuda stream.");
  }
  RUNTIME_PROFILE_LOOP_H2D_D2H_END
  return true;
}

void TrtBackend::GetInputOutputInfo() {
  // Read the original dtypes from inputs_desc_ and outputs_desc_
  std::unordered_map<std::string, FDDataType> inputs_original_dtype_map;
  std::unordered_map<std::string, FDDataType> outputs_original_dtype_map;
  for (size_t i = 0; i < inputs_desc_.size(); ++i) {
    inputs_original_dtype_map[inputs_desc_[i].name] =
        inputs_desc_[i].original_dtype;
  }
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    outputs_original_dtype_map[outputs_desc_[i].name] =
        outputs_desc_[i].original_dtype;
  }

  // Re-read the tensor infos from TRT model and write into inputs_desc_ and
  // outputs_desc_
  std::vector<TrtValueInfo>().swap(inputs_desc_);
  std::vector<TrtValueInfo>().swap(outputs_desc_);
  inputs_desc_.clear();
  outputs_desc_.clear();
  auto num_binds = engine_->getNbBindings();
  for (auto i = 0; i < num_binds; ++i) {
    std::string name = std::string(engine_->getBindingName(i));
    auto shape = ToVec(engine_->getBindingDimensions(i));
    auto dtype = engine_->getBindingDataType(i);
    if (engine_->bindingIsInput(i)) {
      auto original_dtype = inputs_original_dtype_map.count(name)
                                ? inputs_original_dtype_map[name]
                                : GetFDDataType(dtype);
      inputs_desc_.emplace_back(
          TrtValueInfo{name, shape, dtype, original_dtype});
      inputs_device_buffer_[name] = FDDeviceBuffer(dtype);
    } else {
      auto original_dtype = outputs_original_dtype_map.count(name)
                                ? outputs_original_dtype_map[name]
                                : GetFDDataType(dtype);
      outputs_desc_.emplace_back(
          TrtValueInfo{name, shape, dtype, original_dtype});
      outputs_device_buffer_[name] = FDDeviceBuffer(dtype);
      casted_output_tensors_[name] = FDTensor();
    }
    io_name_index_[name] = i;
  }
  bindings_.resize(num_binds);
}

void TrtBackend::SetInputs(const std::vector<FDTensor> &inputs) {
  for (const auto &item : inputs) {
    // auto idx = engine_->getBindingIndex(item.name.c_str());
    auto iter = io_name_index_.find(item.name);
    FDASSERT(iter != io_name_index_.end(),
             "TRTBackend SetInputs not find name:%s", item.name.c_str());
    auto idx = iter->second;
    std::vector<int> shape(item.shape.begin(), item.shape.end());
    auto dims = ToDims(shape);
    context_->setBindingDimensions(idx, dims);

    if (item.device == Device::GPU) {
      if (item.dtype == FDDataType::INT64) {
        inputs_device_buffer_[item.name].resize(dims);
        FDTensor input_tensor;
        input_tensor.SetExternalData(item.shape, FDDataType::INT32,
                                     inputs_device_buffer_[item.name].data(),
                                     Device::GPU);
        function::CudaCast(item, &input_tensor, stream_);
      } else {
        // no copy
        inputs_device_buffer_[item.name].SetExternalData(dims, item.Data());
      }
    } else {
      // Allocate input buffer memory
      inputs_device_buffer_[item.name].resize(dims);

      // copy from cpu to gpu
      if (item.dtype == FDDataType::INT64) {
        int64_t *data = static_cast<int64_t *>(const_cast<void *>(item.Data()));
        std::vector<int32_t> casted_data(data, data + item.Numel());
        // FDASSERT(cudaMemcpyAsync(inputs_device_buffer_[item.name].data(),
        //                          static_cast<void*>(casted_data.data()),
        //                          item.Nbytes() / 2, cudaMemcpyHostToDevice,
        //                          stream_) == 0,
        //          "Error occurs while copy memory from CPU to GPU.");
        // WARN: For cudaMemcpyHostToDevice direction, cudaMemcpyAsync need
        // page-locked host memory to avoid any overlap to occur. The
        // page-locked feature need by cudaMemcpyAsync may not guarantee by
        // FDTensor now. Reference:
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creation-and-destruction
        FDASSERT(cudaMemcpy(inputs_device_buffer_[item.name].data(),
                            static_cast<void *>(casted_data.data()),
                            item.Nbytes() / 2, cudaMemcpyHostToDevice) == 0,
                 "Error occurs while copy memory from CPU to GPU.");
      } else {
        // FDASSERT(cudaMemcpyAsync(inputs_device_buffer_[item.name].data(),
        //                          item.Data(), item.Nbytes(),
        //                          cudaMemcpyHostToDevice, stream_) == 0,
        //          "Error occurs while copy memory from CPU to GPU.");
        // WARN: For cudaMemcpyHostToDevice direction, cudaMemcpyAsync need
        // page-locked host memory to avoid any overlap to occur. The
        // page-locked feature need by cudaMemcpyAsync may not guarantee by
        // FDTensor now. Reference:
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creation-and-destruction
        FDASSERT(cudaMemcpy(inputs_device_buffer_[item.name].data(),
                            item.Data(), item.Nbytes(),
                            cudaMemcpyHostToDevice) == 0,
                 "Error occurs while copy memory from CPU to GPU.");
      }
    }
    // binding input buffer
    bindings_[idx] = inputs_device_buffer_[item.name].data();
  }
}

void TrtBackend::AllocateOutputsBuffer(std::vector<FDTensor> *outputs,
                                       bool copy_to_fd) {
  if (outputs->size() != outputs_desc_.size()) {
    outputs->resize(outputs_desc_.size());
  }
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    // auto idx = engine_->getBindingIndex(outputs_desc_[i].name.c_str());
    auto idx_iter = io_name_index_.find(outputs_desc_[i].name);
    FDASSERT(idx_iter != io_name_index_.end(),
             "TRTBackend Outputs not find name:%s",
             outputs_desc_[i].name.c_str());
    auto idx = idx_iter->second;
    auto output_dims = context_->getBindingDimensions(idx);

    // find the original index of output
    auto iter = outputs_order_.find(outputs_desc_[i].name);
    FDASSERT(
        iter != outputs_order_.end(),
        "Cannot find output: %s of tensorrt network from the original model.",
        outputs_desc_[i].name.c_str());
    auto ori_idx = iter->second;

    // Allocate output buffer memory
    outputs_device_buffer_[outputs_desc_[i].name].resize(output_dims);

    // binding output buffer
    bindings_[idx] = outputs_device_buffer_[outputs_desc_[i].name].data();

    // set user's outputs info
    std::vector<int64_t> shape(output_dims.d,
                               output_dims.d + output_dims.nbDims);
    if (copy_to_fd) {
      (*outputs)[ori_idx].is_pinned_memory = option_.enable_pinned_memory;
      (*outputs)[ori_idx].Resize(shape, outputs_desc_[i].original_dtype,
                                 outputs_desc_[i].name);
    } else {
      (*outputs)[ori_idx].name = outputs_desc_[i].name;
      (*outputs)[ori_idx].SetExternalData(
          shape, outputs_desc_[i].original_dtype, bindings_[idx], Device::GPU,
          option_.gpu_id);
    }
  }
}

bool TrtBackend::BuildTrtEngine() {
  if (option_.enable_log_info) {
    FDTrtLogger::Get()->SetLog(true, true);
  }
  auto config =
      FDUniquePtr<nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());
  if (!config) {
    FDERROR << "Failed to call createBuilderConfig()." << std::endl;
    return false;
  }

  if (option_.enable_fp16) {
    if (!builder_->platformHasFastFp16()) {
      FDWARNING << "Detected FP16 is not supported in the current GPU, "
                   "will use FP32 instead."
                << std::endl;
    } else {
      FDINFO << "[TrtBackend] Use FP16 to inference." << std::endl;
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
  }

  FDINFO << "Start to building TensorRT Engine..." << std::endl;

  if (context_) {
    context_.reset();
    engine_.reset();
  }
  if (option_.max_batch_size >= 1) {
    builder_->setMaxBatchSize(option_.max_batch_size);
  }
  config->setMaxWorkspaceSize(option_.max_workspace_size);
  auto profile = builder_->createOptimizationProfile();
  for (const auto &item : shape_range_info_) {
    FDASSERT(
        profile->setDimensions(item.first.c_str(),
                               nvinfer1::OptProfileSelector::kMIN,
                               ToDims(item.second.min)),
        "[TrtBackend] Failed to set min_shape for input: %s in TrtBackend.",
        item.first.c_str());
    FDASSERT(
        profile->setDimensions(item.first.c_str(),
                               nvinfer1::OptProfileSelector::kMAX,
                               ToDims(item.second.max)),
        "[TrtBackend] Failed to set max_shape for input: %s in TrtBackend.",
        item.first.c_str());
    if (item.second.opt.size() == 0) {
      FDASSERT(
          profile->setDimensions(item.first.c_str(),
                                 nvinfer1::OptProfileSelector::kOPT,
                                 ToDims(item.second.max)),
          "[TrtBackend] Failed to set opt_shape for input: %s in TrtBackend.",
          item.first.c_str());
    } else {
      FDASSERT(
          item.second.opt.size() == item.second.shape.size(),
          "Require the dimension of opt in shape range information equal to "
          "dimension of input: %s in this model, but now it's %zu != %zu.",
          item.first.c_str(), item.second.opt.size(), item.second.shape.size());
      FDASSERT(
          profile->setDimensions(item.first.c_str(),
                                 nvinfer1::OptProfileSelector::kOPT,
                                 ToDims(item.second.opt)),
          "[TrtBackend] Failed to set opt_shape for input: %s in TrtBackend.",
          item.first.c_str());
    }
  }
  config->addOptimizationProfile(profile);

  if (calibration_str_.size()) {
    if (!builder_->platformHasFastInt8()) {
      FDWARNING << "Detected INT8 is not supported in the current GPU, "
                   "will use FP32 instead."
                << std::endl;
    } else {
      FDINFO << "[TrtBackend] Use INT8 to inference." << std::endl;
      config->setFlag(nvinfer1::BuilderFlag::kINT8);
      Int8EntropyCalibrator2 *calibrator =
          new Int8EntropyCalibrator2(calibration_str_);
      config->setInt8Calibrator(calibrator);
    }
  }

  FDUniquePtr<nvinfer1::IHostMemory> plan{
      builder_->buildSerializedNetwork(*network_, *config)};
  if (!plan) {
    FDERROR << "Failed to call buildSerializedNetwork()." << std::endl;
    return false;
  }

  FDUniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(*FDTrtLogger::Get())};
  if (!runtime) {
    FDERROR << "Failed to call createInferRuntime()." << std::endl;
    return false;
  }

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(plan->data(), plan->size()),
      FDInferDeleter());
  if (!engine_) {
    FDERROR << "Failed to call deserializeCudaEngine()." << std::endl;
    return false;
  }

  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  GetInputOutputInfo();

  FDINFO << "TensorRT Engine is built successfully." << std::endl;
  if (option_.serialize_file != "") {
    FDINFO << "Serialize TensorRTEngine to local file "
           << option_.serialize_file << "." << std::endl;
    std::ofstream engine_file(option_.serialize_file.c_str(),
                              std::ios::binary | std::ios::out);
    if (!engine_file) {
      FDERROR << "Failed to open " << option_.serialize_file << " to write."
              << std::endl;
      return false;
    }
    engine_file.write(static_cast<char *>(plan->data()), plan->size());
    engine_file.close();
    FDINFO << "TensorRTEngine is serialized to local file "
           << option_.serialize_file
           << ", we can load this model from the serialized engine "
              "directly next time."
           << std::endl;
  }
  return true;
}

bool TrtBackend::CreateTrtEngineFromOnnx(const std::string &onnx_model_buffer) {
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  builder_ = FDUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(*FDTrtLogger::Get()));
  if (!builder_) {
    FDERROR << "Failed to call createInferBuilder()." << std::endl;
    return false;
  }
  network_ = FDUniquePtr<nvinfer1::INetworkDefinition>(
      builder_->createNetworkV2(explicitBatch));
  if (!network_) {
    FDERROR << "Failed to call createNetworkV2()." << std::endl;
    return false;
  }
  parser_ = FDUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network_, *FDTrtLogger::Get()));
  if (!parser_) {
    FDERROR << "Failed to call createParser()." << std::endl;
    return false;
  }
  bool model_parser;
  if (save_external_) {
    model_parser = !parser_->parseFromFile(onnx_model_buffer.c_str(), 0);
  } else {
    model_parser =
        !parser_->parse(onnx_model_buffer.data(), onnx_model_buffer.size());
  }
  if (model_parser) {
    FDERROR << "Failed to parse ONNX model by TensorRT." << std::endl;
    return false;
  }

  if (option_.serialize_file != "") {
    std::ifstream fin(option_.serialize_file, std::ios::binary | std::ios::in);
    if (fin) {
      FDINFO << "Detect serialized TensorRT Engine file in "
             << option_.serialize_file << ", will load it directly."
             << std::endl;
      fin.close();
      // clear memory buffer of the temporary member
      std::string().swap(onnx_model_buffer_);
      return LoadTrtCache(option_.serialize_file);
    }
  }

  if (!CanBuildEngine(shape_range_info_)) {
    onnx_model_buffer_ = onnx_model_buffer;
    FDWARNING << "Cannot build engine right now, because there's dynamic input "
                 "shape exists, list as below,"
              << std::endl;
    for (int i = 0; i < NumInputs(); ++i) {
      FDWARNING << "Input " << i << ": " << GetInputInfo(i) << std::endl;
    }
    FDWARNING
        << "UltraInfer will build the engine while inference with input data, "
           "and will also collect the input shape range information. You "
           "should be noticed that UltraInfer will rebuild the engine while "
           "new input shape is out of the collected shape range, this may "
           "bring some time consuming problem, refer "
           "https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/en/"
           "faq/"
           "tensorrt_tricks.md for more details."
        << std::endl;
    initialized_ = true;
    return true;
  }

  if (!BuildTrtEngine()) {
    FDERROR << "Failed to build tensorrt engine." << std::endl;
  }

  // clear memory buffer of the temporary member
  std::string().swap(onnx_model_buffer_);
  return true;
}

TensorInfo TrtBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  TensorInfo info;
  info.name = inputs_desc_[index].name;
  info.shape.assign(inputs_desc_[index].shape.begin(),
                    inputs_desc_[index].shape.end());
  info.dtype = inputs_desc_[index].original_dtype;
  return info;
}

std::vector<TensorInfo> TrtBackend::GetInputInfos() {
  std::vector<TensorInfo> infos;
  for (auto i = 0; i < inputs_desc_.size(); i++) {
    infos.emplace_back(GetInputInfo(i));
  }
  return infos;
}

TensorInfo TrtBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs: %d.", index,
           NumOutputs());
  TensorInfo info;
  info.name = outputs_desc_[index].name;
  info.shape.assign(outputs_desc_[index].shape.begin(),
                    outputs_desc_[index].shape.end());
  info.dtype = outputs_desc_[index].original_dtype;
  return info;
}

std::vector<TensorInfo> TrtBackend::GetOutputInfos() {
  std::vector<TensorInfo> infos;
  for (auto i = 0; i < outputs_desc_.size(); i++) {
    infos.emplace_back(GetOutputInfo(i));
  }
  return infos;
}

std::unique_ptr<BaseBackend> TrtBackend::Clone(RuntimeOption &runtime_option,
                                               void *stream, int device_id) {
  std::unique_ptr<BaseBackend> new_backend = utils::make_unique<TrtBackend>();
  auto casted_backend = dynamic_cast<TrtBackend *>(new_backend.get());
  if (device_id > 0 && device_id != option_.gpu_id) {
    auto clone_option = option_;
    clone_option.gpu_id = device_id;
    clone_option.external_stream_ = stream;
    if (runtime_option.model_from_memory_) {
      FDASSERT(casted_backend->InitFromPaddle(runtime_option.model_file,
                                              runtime_option.params_file,
                                              clone_option),
               "Clone model from Paddle failed while initialize TrtBackend.");
    } else {
      if (option_.model_format == ModelFormat::ONNX) {
        std::string model_buffer = "";
        FDASSERT(
            ReadBinaryFromFile(clone_option.model_file, &model_buffer),
            "Fail to read binary from model file while cloning TrtBackend");
        FDASSERT(casted_backend->InitFromOnnx(model_buffer, clone_option),
                 "Clone model from ONNX failed while initialize TrtBackend.");
      } else {
        std::string model_buffer = "";
        std::string params_buffer = "";
        FDASSERT(
            ReadBinaryFromFile(clone_option.model_file, &model_buffer),
            "Fail to read binary from model file while cloning TrtBackend");
        FDASSERT(
            ReadBinaryFromFile(clone_option.params_file, &params_buffer),
            "Fail to read binary from parameter file while cloning TrtBackend");
        FDASSERT(casted_backend->InitFromPaddle(model_buffer, params_buffer,
                                                clone_option),
                 "Clone model from Paddle failed while initialize TrtBackend.");
      }
    }
    FDWARNING << "The target device id:" << device_id
              << " is different from current device id:" << option_.gpu_id
              << ", cannot share memory with current engine." << std::endl;
    return new_backend;
  }
  cudaSetDevice(option_.gpu_id);
  casted_backend->option_.gpu_id = option_.gpu_id;
  if (stream) {
    casted_backend->stream_ = reinterpret_cast<cudaStream_t>(stream);
  } else {
    FDASSERT(cudaStreamCreate(&casted_backend->stream_) == 0,
             "[ERROR] Error occurs while clone calling cudaStreamCreate().");
  }
  casted_backend->inputs_desc_.assign(inputs_desc_.begin(), inputs_desc_.end());
  casted_backend->outputs_desc_.assign(outputs_desc_.begin(),
                                       outputs_desc_.end());
  casted_backend->outputs_order_.insert(outputs_order_.begin(),
                                        outputs_order_.end());
  casted_backend->shape_range_info_.insert(shape_range_info_.begin(),
                                           shape_range_info_.end());
  casted_backend->engine_ = engine_;
  casted_backend->context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      casted_backend->engine_->createExecutionContext());
  casted_backend->GetInputOutputInfo();
  FDINFO << "TRTBackend clone finish." << std::endl;
  return new_backend;
}

} // namespace ultra_infer
