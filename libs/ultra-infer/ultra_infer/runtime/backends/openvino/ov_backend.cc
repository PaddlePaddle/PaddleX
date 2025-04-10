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

#include "ultra_infer/runtime/backends/openvino/ov_backend.h"
#ifdef ENABLE_PADDLE2ONNX
#include "paddle2onnx/converter.h"
#endif

namespace ultra_infer {

std::vector<int64_t> PartialShapeToVec(const ov::PartialShape &shape) {
  std::vector<int64_t> res;
  for (int i = 0; i < shape.size(); ++i) {
    auto dim = shape[i];
    if (dim.is_dynamic()) {
      res.push_back(-1);
    } else {
      res.push_back(dim.get_length());
    }
  }
  return res;
}

ov::PartialShape VecToPartialShape(const std::vector<int64_t> &shape) {
  std::vector<ov::Dimension> dims;
  for (size_t i = 0; i < shape.size(); ++i) {
    dims.emplace_back(ov::Dimension(shape[i]));
  }
  return ov::PartialShape(dims);
}

FDDataType OpenVINODataTypeToFD(const ov::element::Type &type) {
  if (type == ov::element::f32) {
    return FDDataType::FP32;
  } else if (type == ov::element::f16) {
    return FDDataType::FP16;
  } else if (type == ov::element::f64) {
    return FDDataType::FP64;
  } else if (type == ov::element::i8) {
    return FDDataType::INT8;
  } else if (type == ov::element::u8) {
    return FDDataType::UINT8;
  } else if (type == ov::element::i32) {
    return FDDataType::INT32;
  } else if (type == ov::element::i64) {
    return FDDataType::INT64;
  } else {
    FDASSERT(false, "Only support float/double/int8/int32/int64/float16 now.");
  }
  return FDDataType::FP32;
}

ov::element::Type FDDataTypeToOV(const FDDataType &type) {
  if (type == FDDataType::FP32) {
    return ov::element::f32;
  } else if (type == FDDataType::FP64) {
    return ov::element::f64;
  } else if (type == FDDataType::INT8) {
    return ov::element::i8;
  } else if (type == FDDataType::UINT8) {
    return ov::element::u8;
  } else if (type == FDDataType::INT32) {
    return ov::element::i32;
  } else if (type == FDDataType::INT64) {
    return ov::element::i64;
  } else if (type == FDDataType::FP16) {
    return ov::element::f16;
  }
  FDASSERT(false,
           "Only support float/double/int8/uint8/int32/int64/float16 now.");
  return ov::element::f32;
}

ov::Core OpenVINOBackend::core_;

void OpenVINOBackend::InitTensorInfo(
    const std::vector<ov::Output<ov::Node>> &ov_outputs,
    std::map<std::string, TensorInfo> *tensor_infos) {
  for (size_t i = 0; i < ov_outputs.size(); ++i) {
    TensorInfo info;
    auto partial_shape = PartialShapeToVec(ov_outputs[i].get_partial_shape());
    info.shape.assign(partial_shape.begin(), partial_shape.end());
    info.dtype = OpenVINODataTypeToFD(ov_outputs[i].get_element_type());
    auto names = ov_outputs[i].get_names();
    for (const auto &name : names) {
      info.name = name;
      tensor_infos->insert(std::make_pair(info.name, info));
    }
  }
}

bool OpenVINOBackend::Init(const RuntimeOption &option) {
  if (option.model_from_memory_) {
    FDERROR << "OpenVINOBackend doesn't support load model from memory, please "
               "load model from disk."
            << std::endl;
    return false;
  }
  if (option.device != Device::CPU) {
    FDERROR << "OpenVINOBackend only supports Device::CPU, but now its "
            << option.device << "." << std::endl;
    return false;
  }

  if (option.model_format == ModelFormat::PADDLE) {
    return InitFromPaddle(option.model_file, option.params_file,
                          option.openvino_option);
  } else if (option.model_format == ModelFormat::ONNX) {
    return InitFromOnnx(option.model_file, option.openvino_option);
  } else {
    FDERROR << "OpenVINOBackend only supports model format Paddle/ONNX, but "
               "now its "
            << option.model_format << std::endl;
    return false;
  }
  return false;
}

bool OpenVINOBackend::InitFromPaddle(const std::string &model_file,
                                     const std::string &params_file,
                                     const OpenVINOBackendOption &option) {
  if (initialized_) {
    FDERROR << "OpenVINOBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  option_ = option;

  std::shared_ptr<ov::Model> model = core_.read_model(model_file, params_file);
  if (option_.shape_infos.size() > 0) {
    std::map<std::string, ov::PartialShape> shape_infos;
    for (const auto &item : option_.shape_infos) {
      shape_infos[item.first] = VecToPartialShape(item.second);
    }
    model->reshape(shape_infos);
  }

  if (option_.device.find("HETERO") != std::string::npos) {
    auto supported_ops = core_.query_model(model, option_.device);
    for (auto &&op : model->get_ops()) {
      auto &affinity = supported_ops[op->get_friendly_name()];
      if (option_.cpu_operators.find(op->description()) !=
          option_.cpu_operators.end()) {
        op->get_rt_info()["affinity"] = "CPU";
      } else {
        op->get_rt_info()["affinity"] = affinity;
      }
    }
  }

  // Get inputs/outputs information from loaded model
  const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
  std::map<std::string, TensorInfo> input_infos;
  InitTensorInfo(inputs, &input_infos);

  const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
  std::map<std::string, TensorInfo> output_infos;
  InitTensorInfo(outputs, &output_infos);

  // OpenVINO model may not keep the same order with original model
  // So here will reorder it's inputs and outputs
  std::string model_content;
  ReadBinaryFromFile(model_file, &model_content);
  auto reader =
      paddle2onnx::PaddleReader(model_content.c_str(), model_content.size());
  if (reader.num_inputs != input_infos.size()) {
    FDERROR << "The number of inputs from PaddleReader:" << reader.num_inputs
            << " not equal to the number of inputs from OpenVINO:"
            << input_infos.size() << "." << std::endl;
    return false;
  }
  if (reader.num_outputs != output_infos.size()) {
    FDERROR << "The number of outputs from PaddleReader:" << reader.num_outputs
            << " not equal to the number of outputs from OpenVINO:"
            << output_infos.size() << "." << std::endl;
    return false;
  }
  for (int i = 0; i < reader.num_inputs; ++i) {
    auto iter = input_infos.find(std::string(reader.inputs[i].name));
    if (iter == input_infos.end()) {
      FDERROR << "Cannot find input name:" << reader.inputs[i].name
              << " from OpenVINO model." << std::endl;
      return false;
    }
    input_infos_.push_back(iter->second);
  }
  for (int i = 0; i < reader.num_outputs; ++i) {
    auto iter = output_infos.find(std::string(reader.outputs[i].name));
    if (iter == output_infos.end()) {
      FDERROR << "Cannot find output name:" << reader.outputs[i].name
              << " from OpenVINO model." << std::endl;
      return false;
    }
    output_infos_.push_back(iter->second);
  }

  ov::AnyMap properties;
  if (option_.hint == "UNDEFINED") {
    if (option_.device == "CPU" && option_.cpu_thread_num > 0) {
      properties["INFERENCE_NUM_THREADS"] = option_.cpu_thread_num;
    }
    if (option_.num_streams == -1) {
      properties["NUM_STREAMS"] = ov::streams::AUTO;
    } else if (option_.num_streams == -2) {
      properties["NUM_STREAMS"] = ov::streams::NUMA;
    } else if (option_.num_streams > 0) {
      properties["NUM_STREAMS"] = option_.num_streams;
    }

    FDINFO << "number of streams:" << option_.num_streams << "." << std::endl;
    if (option_.affinity == "NO") {
      properties.emplace(ov::hint::enable_cpu_pinning(false));
    }
    FDINFO << "affinity:" << option_.affinity << "." << std::endl;
  } else if (option_.hint == "LATENCY") {
    properties.emplace(
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  } else if (option_.hint == "THROUGHPUT") {
    properties.emplace(
        ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
  } else if (option_.hint == "CUMULATIVE_THROUGHPUT") {
    properties.emplace(ov::hint::performance_mode(
        ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
  }

  FDINFO << "Compile OpenVINO model on device_name:" << option.device << "."
         << std::endl;

  compiled_model_ = core_.compile_model(model, option.device, properties);

  request_ = compiled_model_.create_infer_request();
  initialized_ = true;
  return true;
}

TensorInfo OpenVINOBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of outputs: %d.", index,
           NumOutputs());
  return input_infos_[index];
}

std::vector<TensorInfo> OpenVINOBackend::GetInputInfos() {
  return input_infos_;
}

std::vector<TensorInfo> OpenVINOBackend::GetOutputInfos() {
  return output_infos_;
}

TensorInfo OpenVINOBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs: %d.", index,
           NumOutputs());
  return output_infos_[index];
}

bool OpenVINOBackend::InitFromOnnx(const std::string &model_file,
                                   const OpenVINOBackendOption &option) {
  if (initialized_) {
    FDERROR << "OpenVINOBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  option_ = option;

  std::shared_ptr<ov::Model> model = core_.read_model(model_file);
  if (option_.shape_infos.size() > 0) {
    std::map<std::string, ov::PartialShape> shape_infos;
    for (const auto &item : option_.shape_infos) {
      shape_infos[item.first] = VecToPartialShape(item.second);
    }
    model->reshape(shape_infos);
  }

  if (option_.device.find("HETERO") != std::string::npos) {
    auto supported_ops = core_.query_model(model, option_.device);
    for (auto &&op : model->get_ops()) {
      auto &affinity = supported_ops[op->get_friendly_name()];
      if (option_.cpu_operators.find(op->description()) !=
          option_.cpu_operators.end()) {
        op->get_rt_info()["affinity"] = "CPU";
      } else {
        op->get_rt_info()["affinity"] = affinity;
      }
    }
  }

  // Get inputs/outputs information from loaded model
  const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
  std::map<std::string, TensorInfo> input_infos;
  InitTensorInfo(inputs, &input_infos);

  const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
  std::map<std::string, TensorInfo> output_infos;
  InitTensorInfo(outputs, &output_infos);

  // OpenVINO model may not keep the same order with original model
  // So here will reorder it's inputs and outputs
  std::string model_content;
  ReadBinaryFromFile(model_file, &model_content);
  auto reader =
      paddle2onnx::OnnxReader(model_content.c_str(), model_content.size());
  if (reader.num_inputs != input_infos.size()) {
    FDWARNING << "The number of input names from OnnxReader:"
              << reader.num_outputs
              << " not equal to the number of input names from OpenVINO:"
              << output_infos.size() << "." << std::endl;
  }
  if (reader.num_inputs != inputs.size()) {
    FDERROR << "The number of inputs from OnnxReader:" << reader.num_inputs
            << " not equal to the number of inputs from OpenVINO:"
            << inputs.size() << "." << std::endl;
    return false;
  }
  if (reader.num_outputs != output_infos.size()) {
    FDWARNING << "The number of output names from OnnxReader:"
              << reader.num_outputs
              << " not equal to the number of output names from OpenVINO:"
              << output_infos.size() << "." << std::endl;
  }
  if (reader.num_outputs != outputs.size()) {
    FDERROR << "The number of outputs from OnnxReader:" << reader.num_outputs
            << " not equal to the number of outputs from OpenVINO:"
            << outputs.size() << "." << std::endl;
    return false;
  }
  for (int i = 0; i < reader.num_inputs; ++i) {
    auto iter = input_infos.find(std::string(reader.inputs[i].name));
    if (iter == input_infos.end()) {
      FDERROR << "Cannot find input name:" << reader.inputs[i].name
              << " from OpenVINO model." << std::endl;
      return false;
    }
    input_infos_.push_back(iter->second);
  }
  for (int i = 0; i < reader.num_outputs; ++i) {
    auto iter = output_infos.find(std::string(reader.outputs[i].name));
    if (iter == output_infos.end()) {
      FDERROR << "Cannot find output name:" << reader.outputs[i].name
              << " from OpenVINO model." << std::endl;
      return false;
    }
    output_infos_.push_back(iter->second);
  }

  ov::AnyMap properties;
  if (option_.hint == "UNDEFINED") {
    if (option_.device == "CPU" && option_.cpu_thread_num > 0) {
      properties["INFERENCE_NUM_THREADS"] = option_.cpu_thread_num;
    }
    if (option_.num_streams == -1) {
      properties["NUM_STREAMS"] = ov::streams::AUTO;
    } else if (option_.num_streams == -2) {
      properties["NUM_STREAMS"] = ov::streams::NUMA;
    } else if (option_.num_streams > 0) {
      properties["NUM_STREAMS"] = option_.num_streams;
    }

    FDINFO << "number of streams:" << option_.num_streams << "." << std::endl;
    if (option_.affinity == "NO") {
      properties.emplace(ov::hint::enable_cpu_pinning(false));
    }
    FDINFO << "affinity:" << option_.affinity << "." << std::endl;
  } else if (option_.hint == "LATENCY") {
    properties.emplace(
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  } else if (option_.hint == "THROUGHPUT") {
    properties.emplace(
        ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
  } else if (option_.hint == "CUMULATIVE_THROUGHPUT") {
    properties.emplace(ov::hint::performance_mode(
        ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
  }

  FDINFO << "Compile OpenVINO model on device_name:" << option.device << "."
         << std::endl;
  compiled_model_ = core_.compile_model(model, option.device, properties);

  request_ = compiled_model_.create_infer_request();

  initialized_ = true;
  return true;
}

int OpenVINOBackend::NumInputs() const { return input_infos_.size(); }

int OpenVINOBackend::NumOutputs() const { return output_infos_.size(); }

bool OpenVINOBackend::Infer(std::vector<FDTensor> &inputs,
                            std::vector<FDTensor> *outputs, bool copy_to_fd) {
  if (inputs.size() != input_infos_.size()) {
    FDERROR << "[OpenVINOBackend] Size of the inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << input_infos_.size() << ")." << std::endl;
    return false;
  }

  RUNTIME_PROFILE_LOOP_H2D_D2H_BEGIN
  for (size_t i = 0; i < inputs.size(); ++i) {
    ov::Shape shape(inputs[i].shape.begin(), inputs[i].shape.end());
    ov::Tensor ov_tensor(FDDataTypeToOV(inputs[i].dtype), shape,
                         inputs[i].Data());
    request_.set_tensor(inputs[i].name, ov_tensor);
  }

  RUNTIME_PROFILE_LOOP_BEGIN(1)
  request_.start_async();
  request_.wait();
  RUNTIME_PROFILE_LOOP_END

  outputs->resize(output_infos_.size());
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    auto out_tensor = request_.get_output_tensor(i);
    auto out_tensor_shape = out_tensor.get_shape();
    std::vector<int64_t> shape(out_tensor_shape.begin(),
                               out_tensor_shape.end());
    if (copy_to_fd) {
      (*outputs)[i].Resize(shape,
                           OpenVINODataTypeToFD(out_tensor.get_element_type()),
                           output_infos_[i].name, Device::CPU);
      memcpy((*outputs)[i].MutableData(), out_tensor.data(),
             (*outputs)[i].Nbytes());
    } else {
      (*outputs)[i].name = output_infos_[i].name;
      (*outputs)[i].SetExternalData(
          shape, OpenVINODataTypeToFD(out_tensor.get_element_type()),
          out_tensor.data(), Device::CPU);
    }
  }
  RUNTIME_PROFILE_LOOP_H2D_D2H_END
  return true;
}

std::unique_ptr<BaseBackend>
OpenVINOBackend::Clone(RuntimeOption &runtime_option, void *stream,
                       int device_id) {
  std::unique_ptr<BaseBackend> new_backend =
      utils::make_unique<OpenVINOBackend>();
  auto casted_backend = dynamic_cast<OpenVINOBackend *>(new_backend.get());
  casted_backend->option_ = option_;
  casted_backend->request_ = compiled_model_.create_infer_request();
  casted_backend->input_infos_.assign(input_infos_.begin(), input_infos_.end());
  casted_backend->output_infos_.assign(output_infos_.begin(),
                                       output_infos_.end());
  return new_backend;
}

} // namespace ultra_infer
