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

#include "ultra_infer/runtime/runtime.h"
#include "ultra_infer/utils/unique_ptr.h"
#include "ultra_infer/utils/utils.h"

namespace ultra_infer {

void RuntimeOption::SetModelPath(const std::string &model_path,
                                 const std::string &params_path,
                                 const ModelFormat &format) {
  model_file = model_path;
  params_file = params_path;
  model_format = format;
  model_from_memory_ = false;
}

void RuntimeOption::SetModelBuffer(const std::string &model_buffer,
                                   const std::string &params_buffer,
                                   const ModelFormat &format) {
  model_file = model_buffer;
  params_file = params_buffer;
  model_format = format;
  model_from_memory_ = true;
}

void RuntimeOption::UseGpu(int gpu_id) {
#if defined(WITH_GPU) || defined(WITH_OPENCL)
  device = Device::GPU;
  device_id = gpu_id;

#if defined(WITH_OPENCL) && defined(ENABLE_LITE_BACKEND)
  paddle_lite_option.device = device;
#endif

#else
  FDWARNING << "The UltraInfer didn't compile with GPU, will force to use CPU."
            << std::endl;
  device = Device::CPU;
#endif
}

void RuntimeOption::UseCpu() { device = Device::CPU; }

void RuntimeOption::UseRKNPU2(ultra_infer::rknpu2::CpuName rknpu2_name,
                              ultra_infer::rknpu2::CoreMask rknpu2_core) {
  rknpu2_option.cpu_name = rknpu2_name;
  rknpu2_option.core_mask = rknpu2_core;
  device = Device::RKNPU;
}

void RuntimeOption::UseHorizon() { device = Device::SUNRISENPU; }

void RuntimeOption::UseTimVX() {
  device = Device::TIMVX;
  paddle_lite_option.device = device;
}

void RuntimeOption::UseKunlunXin(int kunlunxin_id, int l3_workspace_size,
                                 bool locked, bool autotune,
                                 const std::string &autotune_file,
                                 const std::string &precision,
                                 bool adaptive_seqlen, bool enable_multi_stream,
                                 int64_t gm_default_size) {
#ifdef WITH_KUNLUNXIN
  device = Device::KUNLUNXIN;

#ifdef ENABLE_LITE_BACKEND
  paddle_lite_option.device = device;
  paddle_lite_option.device_id = kunlunxin_id;
  paddle_lite_option.kunlunxin_l3_workspace_size = l3_workspace_size;
  paddle_lite_option.kunlunxin_locked = locked;
  paddle_lite_option.kunlunxin_autotune = autotune;
  paddle_lite_option.kunlunxin_autotune_file = autotune_file;
  paddle_lite_option.kunlunxin_precision = precision;
  paddle_lite_option.kunlunxin_adaptive_seqlen = adaptive_seqlen;
  paddle_lite_option.kunlunxin_enable_multi_stream = enable_multi_stream;
  paddle_lite_option.kunlunxin_gm_default_size = gm_default_size;
#endif
#ifdef ENABLE_PADDLE_BACKEND
  paddle_infer_option.device = device;
  paddle_infer_option.xpu_option.kunlunxin_device_id = kunlunxin_id;
  paddle_infer_option.xpu_option.kunlunxin_l3_workspace_size =
      l3_workspace_size;
  paddle_infer_option.xpu_option.kunlunxin_locked = locked;
  paddle_infer_option.xpu_option.kunlunxin_autotune = autotune;
  paddle_infer_option.xpu_option.kunlunxin_autotune_file = autotune_file;
  paddle_infer_option.xpu_option.kunlunxin_precision = precision;
  paddle_infer_option.xpu_option.kunlunxin_adaptive_seqlen = adaptive_seqlen;
  paddle_infer_option.xpu_option.kunlunxin_enable_multi_stream =
      enable_multi_stream;
// paddle_infer_option.xpu_option.kunlunxin_gm_default_size = gm_default_size;
// use paddle_infer_option.xpu_option.SetXpuConfig() for more options.
#endif

#else
  FDWARNING
      << "The UltraInfer didn't compile with KUNLUNXIN, will force to use CPU."
      << std::endl;
  device = Device::CPU;
#endif
}

void RuntimeOption::UseIpu(int device_num, int micro_batch_size,
                           bool enable_pipelining, int batches_per_step) {
#ifdef WITH_IPU
  device = Device::IPU;
  paddle_infer_option.ipu_option.ipu_device_num = device_num;
  paddle_infer_option.ipu_option.ipu_micro_batch_size = micro_batch_size;
  paddle_infer_option.ipu_option.ipu_enable_pipelining = enable_pipelining;
  paddle_infer_option.ipu_option.ipu_batches_per_step = batches_per_step;
// use paddle_infer_option.ipu_option.SetIpuConfig() for more options.
#else
  FDWARNING << "The UltraInfer didn't compile with IPU, will force to use CPU."
            << std::endl;
  device = Device::CPU;
#endif
}

void RuntimeOption::UseAscend(int npu_id) {
  device = Device::ASCEND;
  paddle_lite_option.device = device;
  device_id = npu_id;
}

void RuntimeOption::UseDirectML() { device = Device::DIRECTML; }

void RuntimeOption::UseSophgo() {
  device = Device::SOPHGOTPUD;
  UseSophgoBackend();
}

void RuntimeOption::SetExternalStream(void *external_stream) {
  external_stream_ = external_stream;
}

void RuntimeOption::SetCpuThreadNum(int thread_num) {
  FDASSERT(thread_num > 0, "The thread_num must be greater than 0.");
  cpu_thread_num = thread_num;
  paddle_lite_option.cpu_threads = thread_num;
  ort_option.intra_op_num_threads = thread_num;
  openvino_option.cpu_thread_num = thread_num;
  paddle_infer_option.cpu_thread_num = thread_num;
}

void RuntimeOption::SetOrtGraphOptLevel(int level) {
  FDWARNING << "`RuntimeOption::SetOrtGraphOptLevel` will be removed in "
               "v1.2.0, please modify its member variables directly, e.g "
               "`runtime_option.ort_option.graph_optimization_level = 99`."
            << std::endl;
  std::vector<int> supported_level{-1, 0, 1, 2};
  auto valid_level = std::find(supported_level.begin(), supported_level.end(),
                               level) != supported_level.end();
  FDASSERT(valid_level, "The level must be -1, 0, 1, 2.");
  ort_option.graph_optimization_level = level;
}

// use paddle inference backend
void RuntimeOption::UsePaddleBackend() {
#ifdef ENABLE_PADDLE_BACKEND
  backend = Backend::PDINFER;
#else
  FDASSERT(false, "The UltraInfer didn't compile with Paddle Inference.");
#endif
}

// use onnxruntime backend
void RuntimeOption::UseOrtBackend() {
#ifdef ENABLE_ORT_BACKEND
  backend = Backend::ORT;
#else
  FDASSERT(false, "The UltraInfer didn't compile with OrtBackend.");
#endif
}

// use sophgoruntime backend
void RuntimeOption::UseSophgoBackend() {
#ifdef ENABLE_SOPHGO_BACKEND
  backend = Backend::SOPHGOTPU;
#else
  FDASSERT(false, "The UltraInfer didn't compile with SophgoBackend.");
#endif
}

// use poros backend
void RuntimeOption::UsePorosBackend() {
#ifdef ENABLE_POROS_BACKEND
  backend = Backend::POROS;
#else
  FDASSERT(false, "The UltraInfer didn't compile with PorosBackend.");
#endif
}

void RuntimeOption::UseTrtBackend() {
#ifdef ENABLE_TRT_BACKEND
  backend = Backend::TRT;
#else
  FDASSERT(false, "The UltraInfer didn't compile with TrtBackend.");
#endif
}

void RuntimeOption::UseOpenVINOBackend() {
#ifdef ENABLE_OPENVINO_BACKEND
  backend = Backend::OPENVINO;
#else
  FDASSERT(false, "The UltraInfer didn't compile with OpenVINO.");
#endif
}

void RuntimeOption::UseLiteBackend() {
#ifdef ENABLE_LITE_BACKEND
  backend = Backend::LITE;
#else
  FDASSERT(false, "The UltraInfer didn't compile with Paddle Lite.");
#endif
}

void RuntimeOption::UseHorizonNPUBackend() {
#ifdef ENABLE_HORIZON_BACKEND
  backend = Backend::HORIZONNPU;
#else
  FDASSERT(false, "The UltraInfer didn't compile with horizon");
#endif
}

void RuntimeOption::UseOMBackend() {
#ifdef ENABLE_OM_BACKEND
  backend = Backend::OMONNPU;
#else
  FDASSERT(false, "The FastDeploy didn't compile with npu om");
#endif
}

void RuntimeOption::SetPaddleMKLDNN(bool pd_mkldnn) {
  FDWARNING << "`RuntimeOption::SetPaddleMKLDNN` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`option.paddle_infer_option.enable_mkldnn = true`"
            << std::endl;
  paddle_infer_option.enable_mkldnn = pd_mkldnn;
}

void RuntimeOption::DeletePaddleBackendPass(const std::string &pass_name) {
  FDWARNING
      << "`RuntimeOption::DeletePaddleBackendPass` will be removed in v1.2.0, "
         "please use `option.paddle_infer_option.DeletePass` instead."
      << std::endl;
  paddle_infer_option.DeletePass(pass_name);
}
void RuntimeOption::EnablePaddleLogInfo() {
  FDWARNING << "`RuntimeOption::EnablePaddleLogInfo` will be removed in "
               "v1.2.0, please modify its member variable directly, e.g "
               "`option.paddle_infer_option.enable_log_info = true`"
            << std::endl;
  paddle_infer_option.enable_log_info = true;
}

void RuntimeOption::DisablePaddleLogInfo() {
  FDWARNING << "`RuntimeOption::DisablePaddleLogInfo` will be removed in "
               "v1.2.0, please modify its member variable directly, e.g "
               "`option.paddle_infer_option.enable_log_info = false`"
            << std::endl;
  paddle_infer_option.enable_log_info = false;
}

void RuntimeOption::EnablePaddleToTrt() {
#ifdef ENABLE_PADDLE_BACKEND
  FDWARNING << "`RuntimeOption::EnablePaddleToTrt` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`option.paddle_infer_option.enable_trt = true`"
            << std::endl;
  FDINFO << "While using TrtBackend with EnablePaddleToTrt, UltraInfer will "
            "change to use Paddle Inference Backend."
         << std::endl;
  backend = Backend::PDINFER;
  paddle_infer_option.enable_trt = true;
#else
  FDASSERT(false, "While using TrtBackend with EnablePaddleToTrt, require the "
                  "UltraInfer is compiled with Paddle Inference Backend, "
                  "please rebuild your UltraInfer.");
#endif
}

void RuntimeOption::SetPaddleMKLDNNCacheSize(int size) {
  FDWARNING << "`RuntimeOption::SetPaddleMKLDNNCacheSize` will be removed in "
               "v1.2.0, please modify its member variable directly, e.g "
               "`option.paddle_infer_option.mkldnn_cache_size = size`."
            << std::endl;
  paddle_infer_option.mkldnn_cache_size = size;
}

void RuntimeOption::SetOpenVINODevice(const std::string &name) {
  FDWARNING << "`RuntimeOption::SetOpenVINODevice` will be removed in v1.2.0, "
               "please use `RuntimeOption.openvino_option.SetDeivce(const "
               "std::string&)` instead."
            << std::endl;
  openvino_option.SetDevice(name);
}

void RuntimeOption::EnableLiteFP16() {
  FDWARNING << "`RuntimeOption::EnableLiteFP16` will be removed in v1.2.0, "
               "please modify its member variables directly, e.g "
               "`runtime_option.paddle_lite_option.enable_fp16 = true`"
            << std::endl;
  paddle_lite_option.enable_fp16 = true;
}

void RuntimeOption::DisableLiteFP16() {
  FDWARNING << "`RuntimeOption::EnableLiteFP16` will be removed in v1.2.0, "
               "please modify its member variables directly, e.g "
               "`runtime_option.paddle_lite_option.enable_fp16 = false`"
            << std::endl;
  paddle_lite_option.enable_fp16 = false;
}

void RuntimeOption::EnableLiteInt8() {
  FDWARNING << "RuntimeOption::EnableLiteInt8 is a useless api, this calling "
               "will not bring any effects, and will be removed in v1.2.0. if "
               "you load a quantized model, it will automatically run with "
               "int8 mode; otherwise it will run with float mode."
            << std::endl;
}

void RuntimeOption::DisableLiteInt8() {
  FDWARNING << "RuntimeOption::DisableLiteInt8 is a useless api, this calling "
               "will not bring any effects, and will be removed in v1.2.0. if "
               "you load a quantized model, it will automatically run with "
               "int8 mode; otherwise it will run with float mode."
            << std::endl;
}

void RuntimeOption::SetLitePowerMode(LitePowerMode mode) {
  FDWARNING << "`RuntimeOption::SetLitePowerMode` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`runtime_option.paddle_lite_option.power_mode = 3;`"
            << std::endl;
  paddle_lite_option.power_mode = mode;
}

void RuntimeOption::SetLiteOptimizedModelDir(
    const std::string &optimized_model_dir) {
  FDWARNING
      << "`RuntimeOption::SetLiteOptimizedModelDir` will be removed in v1.2.0, "
         "please modify its member variable directly, e.g "
         "`runtime_option.paddle_lite_option.optimized_model_dir = \"...\"`"
      << std::endl;
  paddle_lite_option.optimized_model_dir = optimized_model_dir;
}

void RuntimeOption::SetLiteSubgraphPartitionPath(
    const std::string &nnadapter_subgraph_partition_config_path) {
  FDWARNING << "`RuntimeOption::SetLiteSubgraphPartitionPath` will be removed "
               "in v1.2.0, please modify its member variable directly, e.g "
               "`runtime_option.paddle_lite_option.nnadapter_subgraph_"
               "partition_config_path = \"...\";` "
            << std::endl;
  paddle_lite_option.nnadapter_subgraph_partition_config_path =
      nnadapter_subgraph_partition_config_path;
}

void RuntimeOption::SetLiteSubgraphPartitionConfigBuffer(
    const std::string &nnadapter_subgraph_partition_config_buffer) {
  FDWARNING
      << "`RuntimeOption::SetLiteSubgraphPartitionConfigBuffer` will be "
         "removed in v1.2.0, please modify its member variable directly, e.g "
         "`runtime_option.paddle_lite_option.nnadapter_subgraph_partition_"
         "config_buffer = ...`"
      << std::endl;
  paddle_lite_option.nnadapter_subgraph_partition_config_buffer =
      nnadapter_subgraph_partition_config_buffer;
}

void RuntimeOption::SetLiteContextProperties(
    const std::string &nnadapter_context_properties) {
  FDWARNING << "`RuntimeOption::SetLiteContextProperties` will be removed in "
               "v1.2.0, please modify its member variable directly, e.g "
               "`runtime_option.paddle_lite_option.nnadapter_context_"
               "properties = ...`"
            << std::endl;
  paddle_lite_option.nnadapter_context_properties =
      nnadapter_context_properties;
}

void RuntimeOption::SetLiteModelCacheDir(
    const std::string &nnadapter_model_cache_dir) {
  FDWARNING
      << "`RuntimeOption::SetLiteModelCacheDir` will be removed in v1.2.0, "
         "please modify its member variable directly, e.g "
         "`runtime_option.paddle_lite_option.nnadapter_model_cache_dir = ...`"
      << std::endl;
  paddle_lite_option.nnadapter_model_cache_dir = nnadapter_model_cache_dir;
}

void RuntimeOption::SetLiteDynamicShapeInfo(
    const std::map<std::string, std::vector<std::vector<int64_t>>>
        &nnadapter_dynamic_shape_info) {
  FDWARNING << "`RuntimeOption::SetLiteDynamicShapeInfo` will be removed in "
               "v1.2.0, please modify its member variable directly, e.g "
               "`runtime_option.paddle_lite_option.paddle_lite_option."
               "nnadapter_dynamic_shape_info = ...`"
            << std::endl;
  paddle_lite_option.nnadapter_dynamic_shape_info =
      nnadapter_dynamic_shape_info;
}

void RuntimeOption::SetLiteMixedPrecisionQuantizationConfigPath(
    const std::string &nnadapter_mixed_precision_quantization_config_path) {
  FDWARNING
      << "`RuntimeOption::SetLiteMixedPrecisionQuantizationConfigPath` will be "
         "removed in v1.2.0, please modify its member variable directly, e.g "
         "`runtime_option.paddle_lite_option.paddle_lite_option.nnadapter_"
         "mixed_precision_quantization_config_path = ...`"
      << std::endl;
  paddle_lite_option.nnadapter_mixed_precision_quantization_config_path =
      nnadapter_mixed_precision_quantization_config_path;
}

void RuntimeOption::SetTrtInputShape(const std::string &input_name,
                                     const std::vector<int32_t> &min_shape,
                                     const std::vector<int32_t> &opt_shape,
                                     const std::vector<int32_t> &max_shape) {
  FDWARNING << "`RuntimeOption::SetTrtInputShape` will be removed in v1.2.0, "
               "please use `RuntimeOption.trt_option.SetShape()` instead."
            << std::endl;
  trt_option.SetShape(input_name, min_shape, opt_shape, max_shape);
}

void RuntimeOption::SetTrtInputData(const std::string &input_name,
                                    const std::vector<float> &min_shape_data,
                                    const std::vector<float> &opt_shape_data,
                                    const std::vector<float> &max_shape_data) {
  FDWARNING << "`RuntimeOption::SetTrtInputData` will be removed in v1.2.0, "
               "please use `RuntimeOption.trt_option.SetInputData()` instead."
            << std::endl;
  trt_option.SetInputData(input_name, min_shape_data, opt_shape_data,
                          max_shape_data);
}

void RuntimeOption::SetTrtMaxWorkspaceSize(size_t max_workspace_size) {
  FDWARNING << "`RuntimeOption::SetTrtMaxWorkspaceSize` will be removed in "
               "v1.2.0, please modify its member variable directly, e.g "
               "`RuntimeOption.trt_option.max_workspace_size = "
            << max_workspace_size << "`." << std::endl;
  trt_option.max_workspace_size = max_workspace_size;
}
void RuntimeOption::SetTrtMaxBatchSize(size_t max_batch_size) {
  FDWARNING << "`RuntimeOption::SetTrtMaxBatchSize` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`RuntimeOption.trt_option.max_batch_size = "
            << max_batch_size << "`." << std::endl;
  trt_option.max_batch_size = max_batch_size;
}

void RuntimeOption::EnableTrtFP16() {
  FDWARNING << "`RuntimeOption::EnableTrtFP16` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`runtime_option.trt_option.enable_fp16 = true;`"
            << std::endl;
  trt_option.enable_fp16 = true;
}

void RuntimeOption::DisableTrtFP16() {
  FDWARNING << "`RuntimeOption::DisableTrtFP16` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`runtime_option.trt_option.enable_fp16 = false;`"
            << std::endl;
  trt_option.enable_fp16 = false;
}

void RuntimeOption::EnablePinnedMemory() { enable_pinned_memory = true; }

void RuntimeOption::DisablePinnedMemory() { enable_pinned_memory = false; }

void RuntimeOption::SetTrtCacheFile(const std::string &cache_file_path) {
  FDWARNING << "`RuntimeOption::SetTrtCacheFile` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`runtime_option.trt_option.serialize_file = \""
            << cache_file_path << "\"." << std::endl;
  trt_option.serialize_file = cache_file_path;
}

void RuntimeOption::SetOpenVINOStreams(int num_streams) {
  FDWARNING << "`RuntimeOption::SetOpenVINOStreams` will be removed in v1.2.0, "
               "please modify its member variable directly, e.g "
               "`runtime_option.openvino_option.num_streams = "
            << num_streams << "`." << std::endl;
  openvino_option.num_streams = num_streams;
}

void RuntimeOption::EnablePaddleTrtCollectShape() {
  FDWARNING << "`RuntimeOption::EnablePaddleTrtCollectShape` will be removed "
               "in v1.2.0, please modify its member variable directly, e.g "
               "runtime_option.paddle_infer_option.collect_trt_shape = true`."
            << std::endl;
  paddle_infer_option.collect_trt_shape = true;
}

void RuntimeOption::DisablePaddleTrtCollectShape() {
  FDWARNING << "`RuntimeOption::DisablePaddleTrtCollectShape` will be removed "
               "in v1.2.0, please modify its member variable directly, e.g "
               "runtime_option.paddle_infer_option.collect_trt_shape = false`."
            << std::endl;
  paddle_infer_option.collect_trt_shape = false;
}

void RuntimeOption::DisablePaddleTrtOPs(const std::vector<std::string> &ops) {
  FDWARNING << "`RuntimeOption::DisablePaddleTrtOps` will be removed in "
               "v.1.20, please use "
               "`runtime_option.paddle_infer_option.DisableTrtOps` instead."
            << std::endl;
  paddle_infer_option.DisableTrtOps(ops);
}

void RuntimeOption::UseTVMBackend() {
#ifdef ENABLE_TVM_BACKEND
  backend = Backend::TVM;
#else
  FDASSERT(false, "The UltraInfer didn't compile with TVMBackend.");
#endif
}

} // namespace ultra_infer
