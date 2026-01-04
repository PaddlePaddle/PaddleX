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

#include "ultra_infer/runtime/backends/poros/poros_backend.h"

#include <sys/time.h>

namespace ultra_infer {

TensorInfo PorosBackend::GetInputInfo(int index) {
  // eager mode cann't obtain input information before infer
  TensorInfo info_input;
  return info_input;
}

TensorInfo PorosBackend::GetOutputInfo(int index) {
  // eager mode cann't obtain output information before infer
  TensorInfo info_output;
  return info_output;
}

std::vector<TensorInfo> PorosBackend::GetInputInfos() {
  // eager mode cann't obtain inputs information before infer
  std::vector<TensorInfo> info_inputs;
  return info_inputs;
}

std::vector<TensorInfo> PorosBackend::GetOutputInfos() {
  // eager mode cann't obtain outputs information before infer
  std::vector<TensorInfo> info_outputs;
  return info_outputs;
}

void PorosBackend::BuildOption(const PorosBackendOption &option) {
  _options.device = (option.device == Device::GPU)
                        ? baidu::mirana::poros::Device::GPU
                        : baidu::mirana::poros::Device::CPU;
  _options.long_to_int = option.long_to_int;
  _options.use_nvidia_tf32 = option.use_nvidia_tf32;
  _options.device_id = option.device_id;
  _options.unconst_ops_thres = option.unconst_ops_thres;
  _options.is_dynamic = option.is_dynamic;
  _options.max_workspace_size = option.max_workspace_size;
  _options.use_fp16 = option.enable_fp16;
  return;
}

bool PorosBackend::Compile(const std::string &model_file,
                           std::vector<std::vector<FDTensor>> &prewarm_tensors,
                           const PorosBackendOption &option) {
  if (initialized_) {
    FDERROR << "PorosBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  BuildOption(option);
  torch::jit::Module mod;
  mod = torch::jit::load(model_file);
  mod.eval();
  if (option.device == Device::GPU) {
    mod.to(at::kCUDA);
  } else {
    mod.to(at::kCPU);
  }
  // get inputs_nums and outputs_nums
  auto graph = mod.get_method("forward").graph();
  auto inputs = graph->inputs();
  // remove self node
  _numinputs = inputs.size() - 1;
  // FDTensor to at::Tensor
  std::vector<std::vector<c10::IValue>> prewarm_datas;
  bool is_backend_cuda = (option.device == Device::GPU);
  for (size_t i = 0; i < prewarm_tensors.size(); ++i) {
    std::vector<c10::IValue> prewarm_data;
    for (size_t j = 0; j < prewarm_tensors[i].size(); ++j) {
      auto tensor = CreatePorosValue(prewarm_tensors[i][j], is_backend_cuda);
      prewarm_data.push_back(tensor);
    }
    prewarm_datas.push_back(prewarm_data);
  }
  // get outputs nums
  auto temp_result = mod.forward(prewarm_datas[0]);
  size_t outputs_nums = 0;
  if (temp_result.isTensor()) {
    outputs_nums += 1;
  } else if (temp_result.isTuple()) {
    auto temp_result_tuple = temp_result.toTuple();
    for (size_t i = 0; i < temp_result_tuple->elements().size(); ++i) {
      auto poros_tensor = temp_result_tuple->elements()[i];
      if (poros_tensor.isTensor()) {
        outputs_nums += 1;
      } else if (poros_tensor.isList()) {
        auto poros_tensor_list = poros_tensor.toList();
        outputs_nums += poros_tensor_list.size();
      } else if (poros_tensor.isTuple()) {
        auto poros_tensor_tuple = poros_tensor.toTuple();
        outputs_nums += poros_tensor_tuple->elements().size();
      } else {
        continue;
      }
    }
  }
  _numoutputs = outputs_nums;
  _poros_module = baidu::mirana::poros::Compile(mod, prewarm_datas, _options);
  if (_poros_module == nullptr) {
    FDERROR << "PorosBackend initlize Failed, try initialize again."
            << std::endl;
    return false;
  }
  initialized_ = true;
  return true;
}

bool PorosBackend::Infer(std::vector<FDTensor> &inputs,
                         std::vector<FDTensor> *outputs, bool copy_to_fd) {
  // Convert FD Tensor to PyTorch Tensor
  std::vector<torch::jit::IValue> poros_inputs;
  bool is_backend_cuda =
      _options.device == baidu::mirana::poros::Device::GPU ? true : false;
  for (size_t i = 0; i < inputs.size(); ++i) {
    poros_inputs.push_back(CreatePorosValue(inputs[i], is_backend_cuda));
  }
  // Infer
  auto poros_outputs = _poros_module->forward(poros_inputs);
  // Convert PyTorch Tensor to FD Tensor
  if (poros_outputs.isTensor()) {
    CopyTensorToCpu(poros_outputs.toTensor(), &((*outputs)[0]),
                    is_backend_cuda);
  } else if (poros_outputs.isTuple()) {
    // deal with multi outputs
    auto poros_outputs_tuple = poros_outputs.toTuple();
    size_t index = 0;
    for (size_t i = 0; i < poros_outputs_tuple->elements().size(); ++i) {
      auto poros_tensor = poros_outputs_tuple->elements()[i];
      if (poros_tensor.isTensor()) {
        CopyTensorToCpu(poros_tensor.toTensor(), &((*outputs)[index]),
                        is_backend_cuda);
        index += 1;
      } else if (poros_tensor.isList()) {
        auto poros_tensor_list = poros_tensor.toList();
        for (const auto list_idx : c10::irange(0, poros_tensor_list.size())) {
          const auto &elt = poros_tensor_list.get(list_idx);
          CopyTensorToCpu(elt.toTensor(), &((*outputs)[index]),
                          is_backend_cuda);
          index += 1;
        }
      } else if (poros_tensor.isTuple()) {
        auto poros_tensor_tuple = poros_tensor.toTuple();
        for (size_t j = 0; j < poros_tensor_tuple->elements().size(); ++j) {
          CopyTensorToCpu(poros_tensor_tuple->elements()[j].toTensor(),
                          &((*outputs)[index]), is_backend_cuda);
          index += 1;
        }
      } else {
        continue;
      }
    }
  } else {
    FDERROR << "Convert to FDTensor Failed!!!!!" << std::endl;
  }
  return true;
}

} // namespace ultra_infer
