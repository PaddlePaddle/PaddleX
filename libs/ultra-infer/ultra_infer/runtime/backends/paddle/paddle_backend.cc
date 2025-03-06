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

#include "ultra_infer/runtime/backends/paddle/paddle_backend.h"

#include <sstream>

#include "ultra_infer/utils/path.h"

namespace ultra_infer {

void PaddleBackend::BuildOption(const PaddleBackendOption &option) {
  option_ = option;
  if (option.device == Device::GPU) {
    auto inference_precision = paddle_infer::PrecisionType::kFloat32;
    if (option_.inference_precision == "float32") {
      FDINFO << "Will inference_precision float32" << std::endl;
      inference_precision = paddle_infer::PrecisionType::kFloat32;
    } else if (option_.inference_precision == "float16") {
      FDINFO << "Will inference_precision float16" << std::endl;
      inference_precision = paddle_infer::PrecisionType::kHalf;
    } else if (option_.inference_precision == "bfloat16") {
      FDINFO << "Will inference_precision bfloat16" << std::endl;
      inference_precision = paddle_infer::PrecisionType::kBf16;
    } else if (option_.inference_precision == "int8") {
      FDINFO << "Will inference_precision int8" << std::endl;
      inference_precision = paddle_infer::PrecisionType::kInt8;
    } else {
      FDERROR << "paddle inference only support precision in float32,"
              << " float16, bfloat16 and int8" << std::endl;
    }
    config_.Exp_DisableMixedPrecisionOps({"feed", "fetch"});
    config_.EnableUseGpu(option.gpu_mem_init_size, option.device_id,
                         inference_precision);
    // config_.EnableUseGpu(option.gpu_mem_init_size, option.device_id);
    if (option_.switch_ir_debug) {
      FDINFO << "Will Enable ir_debug for Paddle Backend." << std::endl;
      config_.SwitchIrDebug();
    }
    if (option_.enable_inference_cutlass) {
#ifdef PADDLEINFERENCE_API_COMPAT_2_4_x
      FDWARNING
          << "Your are using Paddle infernence 2.4.x, cutlass is not supported!"
          << std::endl;
#else
      FDINFO << "Will enable_inference_cutlass" << std::endl;
      config_.Exp_EnableUseCutlass();
#endif
    }
    if (option_.external_stream_) {
      FDINFO << "Will use external stream for Paddle Backend." << std::endl;
      config_.SetExecStream(option_.external_stream_);
    }
    if (option.enable_trt) {
      if (!option.trt_option.enable_fp16) {
        FDINFO << "Will try to use tensorrt inference with Paddle Backend."
               << std::endl;
      }
      config_.Exp_DisableTensorRtOPs(option.trt_disabled_ops_);
      auto precision = paddle_infer::PrecisionType::kFloat32;
      if (option.trt_option.enable_fp16) {
        FDINFO << "Will try to use tensorrt fp16 inference with Paddle Backend."
               << std::endl;
        precision = paddle_infer::PrecisionType::kHalf;
      }
      bool use_static = false;
      if (option.trt_option.serialize_file != "") {
        FDWARNING
            << "Detect that tensorrt cache file has been set to "
            << option.trt_option.serialize_file
            << ", but while enable paddle2trt, please notice that the cache "
               "file will save to the directory where paddle model saved."
            << std::endl;
        use_static = true;
        std::string opt_cache_dir =
            GetDirFromPath(option.trt_option.serialize_file);

        config_.SetOptimCacheDir(opt_cache_dir);
      }
      config_.EnableTensorRtEngine(option.trt_option.max_workspace_size,
                                   option.trt_option.max_batch_size,
                                   option.trt_min_subgraph_size, precision,
                                   use_static);

      if (!option.collect_trt_shape) {
        SetTRTDynamicShapeToConfig(option);
      }
      if (option_.enable_fixed_size_opt) {
        paddle_infer::experimental::InternalUtils::SetTransformerMaskid(
            &config_, "opt");
      }
    }
  } else if (option.device == Device::IPU) {
#ifdef WITH_IPU
    config_.EnableIpu(option.ipu_option.ipu_device_num,
                      option.ipu_option.ipu_micro_batch_size,
                      option.ipu_option.ipu_enable_pipelining,
                      option.ipu_option.ipu_batches_per_step);
    config_.SetIpuConfig(option.ipu_option.ipu_enable_fp16,
                         option.ipu_option.ipu_replica_num,
                         option.ipu_option.ipu_available_memory_proportion,
                         option.ipu_option.ipu_enable_half_partial);
#else
    FDWARNING << "The UltraInfer is not compiled with IPU device, so will "
                 "fallback to CPU with Paddle Inference Backend."
              << std::endl;
#endif
  } else if (option.device == Device::KUNLUNXIN) {
#ifdef WITH_KUNLUNXIN
    // Note(qiuyanjun): For Paddle XPU L3 Cache, please set
    // export XPU_PADDLE_L3_SIZE=67104768 (XPU R200)
    // export FLAGS_fuse_multi_transformer_quant_type="float"
    config_.EnableXpu(option.xpu_option.kunlunxin_l3_workspace_size,
                      option.xpu_option.kunlunxin_locked,
                      option.xpu_option.kunlunxin_autotune,
                      option.xpu_option.kunlunxin_autotune_file,
                      option.xpu_option.kunlunxin_precision,
                      option.xpu_option.kunlunxin_adaptive_seqlen,
                      option.xpu_option.kunlunxin_enable_multi_stream);
    config_.SetXpuConfig(
        option.xpu_option.kunlunxin_quant_post_dynamic_weight_bits,
        option.xpu_option.kunlunxin_quant_post_dynamic_op_types);
    config_.SetXpuDeviceId(option.xpu_option.kunlunxin_device_id);
#else
    FDWARNING
        << "The UltraInfer is not compiled with KUNLUNXIN device, so will "
           "fallback to CPU with Paddle Inference Backend."
        << std::endl;
#endif
  } else {
    config_.DisableGpu();
    if (option.enable_mkldnn) {
      config_.EnableMKLDNN();
      config_.SetMkldnnCacheCapacity(option.mkldnn_cache_size);
    } else {
#if defined(PADDLEINFERENCE_API_COMPAT_2_6_x) ||                               \
    (PADDLEINFERENCE_VERSION_MAJOR != 2)
      config_.DisableMKLDNN();
#endif
    }
  }

  if (!option.enable_log_info) {
    config_.DisableGlogInfo();
  }
  if (option.cpu_thread_num <= 0) {
    config_.SetCpuMathLibraryNumThreads(8);
  } else {
    config_.SetCpuMathLibraryNumThreads(option.cpu_thread_num);
  }
  // Note: SwitchIrOptim is enabled by default for paddle inference
  // backend. So, we don't need to set it manually.
  // config_.SwitchIrOptim(option.switch_ir_optimize);

  if (option.enable_new_ir) {
#if PADDLEINFERENCE_VERSION_MAJOR == 2
    FDWARNING << "UltraInfer was compiled with Paddle Inference v2.0+ "
                 "which does not support the new IR."
              << std::endl;
#else
    if (option.device == Device::GPU && option.enable_trt) {
      FDWARNING << "Currently, Paddle-TensorRT does not support the new IR, "
                   "and the old IR will be used."
                << std::endl;
    } else {
      config_.EnableNewIR();
      config_.EnableNewExecutor();
      if (option.device == Device::CPU || option.device == Device::GPU) {
        config_.SetOptimizationLevel(3);
      }
    }
#endif
  }
}

bool PaddleBackend::Init(const RuntimeOption &runtime_option) {
  if (!(Supported(runtime_option.model_format, Backend::PDINFER) &&
        Supported(runtime_option.device, Backend::PDINFER))) {
    return false;
  }

  auto option = runtime_option;
  // Collect basic paddle inference option and trt option.
  option.paddle_infer_option.model_file = runtime_option.model_file;
  option.paddle_infer_option.params_file = runtime_option.params_file;
  option.paddle_infer_option.model_from_memory_ =
      runtime_option.model_from_memory_;
  option.paddle_infer_option.device = runtime_option.device;
  option.paddle_infer_option.device_id = runtime_option.device_id;
  option.paddle_infer_option.enable_pinned_memory =
      runtime_option.enable_pinned_memory;
  option.paddle_infer_option.external_stream_ = runtime_option.external_stream_;
  option.paddle_infer_option.trt_option = runtime_option.trt_option;
  option.paddle_infer_option.trt_option.gpu_id = runtime_option.device_id;
  // Note(qiuyanjun): For Ipu option and XPU option, please check the
  // details of RuntimeOption::UseIpu() and RuntimeOption::UseKunlunXin().
  // Futhermore, please check paddle_infer_option.SetIpuConfig() and
  // paddle_infer_option.SetXpuConfig() for more details of extra configs.
  return InitFromPaddle(option.model_file, option.params_file,
                        option.model_from_memory_, option.paddle_infer_option);
}

bool PaddleBackend::InitFromPaddle(const std::string &model,
                                   const std::string &params,
                                   bool model_from_memory,
                                   const PaddleBackendOption &option) {
  if (initialized_) {
    FDERROR << "PaddleBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  if (model_from_memory) {
    config_.SetModelBuffer(model.c_str(), model.size(), params.c_str(),
                           params.size());
  } else {
    config_.SetModel(model, params);
  }
  if (option.enable_memory_optimize) {
    config_.EnableMemoryOptim();
  }
  BuildOption(option);
  // The input/output information get from predictor is not right, use
  // PaddleReader instead now
  std::string model_content = model;
  if (!model_from_memory) {
    FDASSERT(ReadBinaryFromFile(model, &model_content),
             "Failed to read file %s.", model.c_str());
  }

  if (option.is_quantize_model) {
    if (option.device == Device::GPU) {
      FDWARNING << "The loaded model is a quantized model, while inference on "
                   "GPU, please use TensorRT backend to get better performance."
                << std::endl;
      if (option.enable_trt) {
        bool use_static = false;
        if (option.trt_option.serialize_file != "") {
          FDWARNING
              << "Detect that tensorrt cache file has been set to "
              << option.trt_option.serialize_file
              << ", but while enable paddle2trt, please notice that the cache "
                 "file will save to the directory where paddle model saved."
              << std::endl;
          use_static = true;
        }
#if PADDLEINFERENCE_VERSION_MAJOR != 2
        config_.EnableTensorRtEngine(
            option.trt_option.max_workspace_size,
            option.trt_option.max_batch_size, option.trt_min_subgraph_size,
            paddle_infer::PrecisionType::kInt8, use_static, false, true);
#else
        config_.EnableTensorRtEngine(
            option.trt_option.max_workspace_size,
            option.trt_option.max_batch_size, option.trt_min_subgraph_size,
            paddle_infer::PrecisionType::kInt8, use_static, false);
#endif
        SetTRTDynamicShapeToConfig(option);
      }
    }
    if (option.enable_mkldnn) {
      config_.EnableMkldnnInt8();
    } else {
      FDWARNING << "The loaded model is a quantized model, while inference on "
                   "CPU, please enable MKLDNN to get better performance."
                << std::endl;
    }
  }
  if (option.collect_trt_shape) {
    // Set the shape info file.
    std::string curr_model_dir = "./";
    if (!option.model_from_memory_) {
      curr_model_dir = GetDirFromPath(option.model_file);
    }
    std::string shape_range_info =
        PathJoin(curr_model_dir, "shape_range_info.pbtxt");
    if (!CheckFileExists(shape_range_info)) {
      FDINFO << "Start generating shape range info file." << std::endl;
      paddle_infer::Config analysis_config;
      if (model_from_memory) {
        analysis_config.SetModelBuffer(model.c_str(), model.size(),
                                       params.c_str(), params.size());
      } else {
        analysis_config.SetModel(model, params);
      }
      if (option.collect_trt_shape_by_device) {
        if (option.device == Device::GPU) {
          analysis_config.EnableUseGpu(option.gpu_mem_init_size,
                                       option.device_id,
                                       paddle_infer::PrecisionType::kFloat32);
        }
      }
      analysis_config.CollectShapeRangeInfo(shape_range_info);
      auto predictor_tmp = paddle_infer::CreatePredictor(analysis_config);
      std::map<std::string, std::vector<int>> max_shape;
      std::map<std::string, std::vector<int>> min_shape;
      std::map<std::string, std::vector<int>> opt_shape;
      GetDynamicShapeFromOption(option, &max_shape, &min_shape, &opt_shape);
      std::map<std::string, std::vector<float>> max_input_data;
      std::map<std::string, std::vector<float>> min_input_data;
      std::map<std::string, std::vector<float>> opt_input_data;
      if (!option.trt_option.min_input_data.empty()) {
        GetInputDataFromOption(option, &max_input_data, &min_input_data,
                               &opt_input_data);
      }
      // Need to run once to get the shape range info file.
      CollectShapeRun(predictor_tmp.get(), max_shape, max_input_data);
      CollectShapeRun(predictor_tmp.get(), min_shape, min_input_data);
      CollectShapeRun(predictor_tmp.get(), opt_shape, opt_input_data);
      CollectShapeRun(predictor_tmp.get(), opt_shape, opt_input_data);
      FDINFO << "Finish generating shape range info file." << std::endl;
    }
    FDINFO << "Start loading shape range info file " << shape_range_info
           << " to set TensorRT dynamic shape." << std::endl;
    config_.EnableTunedTensorRtDynamicShape(shape_range_info,
                                            option.allow_build_trt_at_runtime);
  }
  // Note(zhoushunjie): The pass deletion should be executed just before
  // creating predictor.
  if (!option.delete_pass_names.empty()) {
    auto pass_builder = config_.pass_builder();
    for (int i = 0; i < option.delete_pass_names.size(); i++) {
      FDINFO << "Delete pass : " << option.delete_pass_names[i] << std::endl;
      pass_builder->DeletePass(option.delete_pass_names[i]);
    }
  }
  if (option.enable_log_info) {
    FDINFO << "Finish paddle inference config with summary as: " << std::endl
           << config_.Summary() << std::endl;
  }
  predictor_ = paddle_infer::CreatePredictor(config_);
  auto input_names = predictor_->GetInputNames();
  auto output_names = predictor_->GetOutputNames();
  auto input_dtypes = predictor_->GetInputTypes();

#ifdef PADDLEINFERENCE_API_COMPAT_2_4_x
  // Note: GetInputTensorShape, GetOutputTensorShape and GetOutputTypes
  // are not supported when Paddle Inference API version is 2.4.x.
  std::map<std::string, std::vector<int64_t>> input_shapes;
  std::map<std::string, std::vector<int64_t>> output_shapes;
  std::map<std::string, paddle_infer::DataType> output_dtypes;
  // Get the all the input shape info.
  for (size_t i = 0; i < input_names.size(); ++i) {
    std::vector<int64_t> shape;
    auto handle = predictor_->GetInputHandle(input_names[i]);
    for (int j = 0; j < handle->shape().size(); ++j) {
      shape.push_back(
          static_cast<int64_t>(handle->shape()[j])); // int32 -> int64
    }
    input_shapes[input_names[i]] = shape;
  }
  // Get the all the output shape and dtype info.
  for (size_t i = 0; i < output_names.size(); ++i) {
    std::vector<int64_t> shape;
    auto handle = predictor_->GetOutputHandle(output_names[i]);
    for (int j = 0; j < handle->shape().size(); ++j) {
      shape.push_back(
          static_cast<int64_t>(handle->shape()[j])); // int32 -> int64
    }
    output_shapes[output_names[i]] = shape;
    output_dtypes[output_names[i]] = handle->type();
  }
#else
  auto input_shapes = predictor_->GetInputTensorShape();
  auto output_shapes = predictor_->GetOutputTensorShape();
  auto output_dtypes = predictor_->GetOutputTypes();
#endif

  inputs_desc_.resize(input_names.size());
  for (int i = 0; i < input_names.size(); ++i) {
    inputs_desc_[i].name = input_names[i];
    auto iter = input_shapes.find(inputs_desc_[i].name);
    FDASSERT(iter != input_shapes.end(), "Cannot find shape for input %s.",
             inputs_desc_[i].name.c_str());
    inputs_desc_[i].shape.assign(iter->second.begin(), iter->second.end());
    auto iter1 = input_dtypes.find(inputs_desc_[i].name);
    FDASSERT(iter1 != input_dtypes.end(), "Cannot find data type for input %s.",
             inputs_desc_[i].name.c_str());
    inputs_desc_[i].dtype = PaddleDataTypeToFD(iter1->second);
  }
  outputs_desc_.resize(output_names.size());
  for (int i = 0; i < output_names.size(); ++i) {
    outputs_desc_[i].name = output_names[i];
    auto iter = output_shapes.find(outputs_desc_[i].name);
    FDASSERT(iter != output_shapes.end(), "Cannot find shape for output %s.",
             outputs_desc_[i].name.c_str());
    outputs_desc_[i].shape.assign(iter->second.begin(), iter->second.end());
    auto iter1 = output_dtypes.find(outputs_desc_[i].name);
    FDASSERT(iter1 != output_dtypes.end(),
             "Cannot find data type for output %s.",
             outputs_desc_[i].name.c_str());
    outputs_desc_[i].dtype = PaddleDataTypeToFD(iter1->second);
  }

  initialized_ = true;
  return true;
}

TensorInfo PaddleBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  return inputs_desc_[index];
}

std::vector<TensorInfo> PaddleBackend::GetInputInfos() { return inputs_desc_; }

TensorInfo PaddleBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs());
  return outputs_desc_[index];
}

std::vector<TensorInfo> PaddleBackend::GetOutputInfos() {
  return outputs_desc_;
}

bool PaddleBackend::Infer(std::vector<FDTensor> &inputs,
                          std::vector<FDTensor> *outputs, bool copy_to_fd) {
  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[PaddleBackend] Size of inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }
  // output share backend memory only support CPU or GPU
  if (option_.device == Device::IPU) {
    copy_to_fd = true;
  }

  RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto handle = predictor_->GetInputHandle(inputs[i].name);
    ShareTensorFromFDTensor(handle.get(), inputs[i]);
  }
  // prebinded output only support for GPU
  // if (!copy_to_fd) {
  //   for (size_t i = 0; i < (*outputs).size(); ++i) {
  //     auto output_name = (*outputs)[i].name;
  //     // if a output is not prebinded,
  //     // the name of output is expected to be empty.
  //     // We skip here
  //     if (output_name.empty()) {
  //       continue;
  //     }
  //     // Record the prebinded output_name.
  //     // Those outputs do not need PaddleTensorToFDTensor
  //     // after predictor_.Run()
  //     auto handle = predictor_->GetOutputHandle(output_name);
  //     ShareOutTensorFromFDTensor(handle.get(), (*outputs)[i]);
  //   }
  // }

  RUNTIME_PROFILE_LOOP_BEGIN(1)
  predictor_->Run();
  RUNTIME_PROFILE_LOOP_END

  outputs->resize(outputs_desc_.size());
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    auto handle = predictor_->GetOutputHandle(outputs_desc_[i].name);
    if (copy_to_fd) {
      (*outputs)[i].is_pinned_memory = option_.enable_pinned_memory;
    }
    PaddleTensorToFDTensor(handle, &((*outputs)[i]), copy_to_fd);
  }
  RUNTIME_PROFILE_LOOP_H2D_D2H_END
  return true;
}

std::unique_ptr<BaseBackend> PaddleBackend::Clone(RuntimeOption &runtime_option,
                                                  void *stream, int device_id) {
  std::unique_ptr<BaseBackend> new_backend =
      utils::make_unique<PaddleBackend>();
  auto casted_backend = dynamic_cast<PaddleBackend *>(new_backend.get());
  if (device_id > 0 && (option_.device == Device::GPU) &&
      device_id != option_.device_id) {
    auto clone_option = option_;
    clone_option.device_id = device_id;
    clone_option.external_stream_ = stream;
    FDASSERT(casted_backend->InitFromPaddle(
                 runtime_option.model_file, runtime_option.params_file,
                 runtime_option.model_from_memory_, clone_option),
             "Clone model from Paddle failed while initialize PaddleBackend.");
    FDWARNING << "The target device id:" << device_id
              << " is different from current device id:" << option_.device_id
              << ", cannot share memory with current engine." << std::endl;
    return new_backend;
  }
  casted_backend->inputs_desc_.assign(inputs_desc_.begin(), inputs_desc_.end());
  casted_backend->outputs_desc_.assign(outputs_desc_.begin(),
                                       outputs_desc_.end());
  casted_backend->predictor_ = std::move(predictor_->Clone(stream));
  return new_backend;
}

void PaddleBackend::SetTRTDynamicShapeToConfig(
    const PaddleBackendOption &option) {
  std::map<std::string, std::vector<int>> max_shape;
  std::map<std::string, std::vector<int>> min_shape;
  std::map<std::string, std::vector<int>> opt_shape;
  GetDynamicShapeFromOption(option, &max_shape, &min_shape, &opt_shape);
  if (min_shape.size() > 0) {
    FDINFO << "Start setting trt dynamic shape." << std::endl;
    config_.SetTRTDynamicShapeInfo(min_shape, max_shape, opt_shape);
    FDINFO << "Finish setting trt dynamic shape." << std::endl;
  }
}

void PaddleBackend::GetDynamicShapeFromOption(
    const PaddleBackendOption &option,
    std::map<std::string, std::vector<int>> *max_shape,
    std::map<std::string, std::vector<int>> *min_shape,
    std::map<std::string, std::vector<int>> *opt_shape) const {
  auto print_shape = [](const std::vector<int> &shape) -> std::string {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < shape.size(); ++i) {
      oss << shape[i];
      if (i < shape.size() - 1) {
        oss << ", ";
      }
    }
    oss << "]";
    return oss.str();
  };
  for (const auto &item : option.trt_option.min_shape) {
    auto max_iter = option.trt_option.max_shape.find(item.first);
    auto opt_iter = option.trt_option.opt_shape.find(item.first);
    FDASSERT(max_iter != option.trt_option.max_shape.end(),
             "Cannot find %s in TrtBackendOption::min_shape.",
             item.first.c_str());
    FDASSERT(opt_iter != option.trt_option.opt_shape.end(),
             "Cannot find %s in TrtBackendOption::opt_shape.",
             item.first.c_str());
    (*max_shape)[item.first].assign(max_iter->second.begin(),
                                    max_iter->second.end());
    (*opt_shape)[item.first].assign(opt_iter->second.begin(),
                                    opt_iter->second.end());
    (*min_shape)[item.first].assign(item.second.begin(), item.second.end());
    FDINFO << item.first
           << ": the max shape = " << print_shape(max_iter->second)
           << ", the min shape = " << print_shape(item.second)
           << ", the opt shape = " << print_shape(opt_iter->second)
           << std::endl;
  }
}

void PaddleBackend::GetInputDataFromOption(
    const PaddleBackendOption &option,
    std::map<std::string, std::vector<float>> *max_input_data,
    std::map<std::string, std::vector<float>> *min_input_data,
    std::map<std::string, std::vector<float>> *opt_input_data) const {
  for (const auto &item : option.trt_option.min_input_data) {
    auto max_iter = option.trt_option.max_input_data.find(item.first);
    auto opt_iter = option.trt_option.opt_input_data.find(item.first);
    FDASSERT(max_iter != option.trt_option.max_input_data.end(),
             "Cannot find %s in TrtBackendOption::min_input_data.",
             item.first.c_str());
    FDASSERT(opt_iter != option.trt_option.opt_input_data.end(),
             "Cannot find %s in TrtBackendOption::opt_input_data.",
             item.first.c_str());
    (*max_input_data)[item.first].assign(max_iter->second.begin(),
                                         max_iter->second.end());
    (*opt_input_data)[item.first].assign(opt_iter->second.begin(),
                                         opt_iter->second.end());
    (*min_input_data)[item.first].assign(item.second.begin(),
                                         item.second.end());
  }
}

void PaddleBackend::CollectShapeRun(
    paddle_infer::Predictor *predictor,
    const std::map<std::string, std::vector<int>> &shape,
    const std::map<std::string, std::vector<float>> &data) const {
  auto input_names = predictor->GetInputNames();
  auto input_type = predictor->GetInputTypes();
  for (const auto &name : input_names) {
    FDASSERT(shape.find(name) != shape.end() &&
                 input_type.find(name) != input_type.end(),
             "When collect_trt_shape is true, please define max/opt/min shape "
             "for model's input:[\"%s\"] by "
             "(C++)RuntimeOption.trt_option.SetShape/"
             "(Python)RuntimeOption.trt_option.set_shape.",
             name.c_str());
    auto tensor = predictor->GetInputHandle(name);
    auto shape_value = shape.at(name);
    int shape_num = std::accumulate(shape_value.begin(), shape_value.end(), 1,
                                    std::multiplies<int>());
    tensor->Reshape(shape_value);

    if (data.find(name) != data.end()) {
      FDASSERT(data.at(name).size() == shape_num,
               "The data num and accumulate of shape must be equal for input: "
               "[\"%s\"], "
               " When Use the (C++)RuntimeOption.trt_option.SetInputData/ "
               " (Python)RuntimeOption.trt_option.set_input_data/",
               name.c_str());
    }

    auto dtype = input_type[name];
    switch (dtype) {
    case paddle_infer::DataType::FLOAT32: {
      if (data.find(name) != data.end()) {
        tensor->CopyFromCpu(data.at(name).data());
      } else {
        std::vector<float> input_data(shape_num, 1.0);
        tensor->CopyFromCpu(input_data.data());
      }
      break;
    }
    case paddle_infer::DataType::INT32: {
      if (data.find(name) != data.end()) {
        std::vector<int> input_data(data.at(name).begin(), data.at(name).end());
        tensor->CopyFromCpu(input_data.data());
      } else {
        std::vector<int> input_data(shape_num, 1);
        tensor->CopyFromCpu(input_data.data());
      }
      break;
    }
    case paddle_infer::DataType::INT64: {
      if (data.find(name) != data.end()) {
        std::vector<int64_t> input_data(data.at(name).begin(),
                                        data.at(name).end());
        tensor->CopyFromCpu(input_data.data());
      } else {
        std::vector<int64_t> input_data(shape_num, 1);
        tensor->CopyFromCpu(input_data.data());
      }
      break;
    }
    default: {
      FDASSERT(false, "Input data Paddle backend only supports "
                      "FP32/INT32/INT64 currently.");
      break;
    }
    }
  }
  predictor->Run();
}

} // namespace ultra_infer
