// Cropyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "ultra_infer/pybind/main.h"

namespace ultra_infer {

void BindLiteOption(pybind11::module &m);
void BindOpenVINOOption(pybind11::module &m);
void BindOrtOption(pybind11::module &m);
void BindTrtOption(pybind11::module &m);
void BindPaddleOption(pybind11::module &m);
void BindPorosOption(pybind11::module &m);
void BindRKNPU2Option(pybind11::module &m);
void BindOption(pybind11::module &m) {
  BindLiteOption(m);
  BindOpenVINOOption(m);
  BindOrtOption(m);
  BindTrtOption(m);
  BindPaddleOption(m);
  BindPorosOption(m);
  BindRKNPU2Option(m);

  pybind11::class_<RuntimeOption>(m, "RuntimeOption")
      .def(pybind11::init())
      .def("set_model_path", &RuntimeOption::SetModelPath)
      .def("set_model_buffer", &RuntimeOption::SetModelBuffer)
      .def("use_gpu", &RuntimeOption::UseGpu)
      .def("use_cpu", &RuntimeOption::UseCpu)
      .def("use_rknpu2", &RuntimeOption::UseRKNPU2)
      .def("use_sophgo", &RuntimeOption::UseSophgo)
      .def("use_ascend", &RuntimeOption::UseAscend)
      .def("use_kunlunxin", &RuntimeOption::UseKunlunXin)
      .def("disable_valid_backend_check",
           &RuntimeOption::DisableValidBackendCheck)
      .def("enable_valid_backend_check",
           &RuntimeOption::EnableValidBackendCheck)
      .def_readwrite("paddle_lite_option", &RuntimeOption::paddle_lite_option)
      .def_readwrite("openvino_option", &RuntimeOption::openvino_option)
      .def_readwrite("ort_option", &RuntimeOption::ort_option)
      .def_readwrite("trt_option", &RuntimeOption::trt_option)
      .def_readwrite("poros_option", &RuntimeOption::poros_option)
      .def_readwrite("paddle_infer_option", &RuntimeOption::paddle_infer_option)
      .def("set_external_stream", &RuntimeOption::SetExternalStream)
      .def("set_external_raw_stream",
           [](RuntimeOption &self, size_t external_stream) {
             self.SetExternalStream(reinterpret_cast<void *>(external_stream));
           })
      .def("set_cpu_thread_num", &RuntimeOption::SetCpuThreadNum)
      .def("use_paddle_backend", &RuntimeOption::UsePaddleBackend)
      .def("use_poros_backend", &RuntimeOption::UsePorosBackend)
      .def("use_tvm_backend", &RuntimeOption::UseTVMBackend)
      .def("use_ort_backend", &RuntimeOption::UseOrtBackend)
      .def("use_trt_backend", &RuntimeOption::UseTrtBackend)
      .def("use_openvino_backend", &RuntimeOption::UseOpenVINOBackend)
      .def("use_lite_backend", &RuntimeOption::UseLiteBackend)
      .def("use_om_backend", &RuntimeOption::UseOMBackend)
      .def("enable_pinned_memory", &RuntimeOption::EnablePinnedMemory)
      .def("disable_pinned_memory", &RuntimeOption::DisablePinnedMemory)
      .def("use_ipu", &RuntimeOption::UseIpu)
      .def("enable_profiling", &RuntimeOption::EnableProfiling)
      .def("disable_profiling", &RuntimeOption::DisableProfiling)
      .def_readwrite("model_file", &RuntimeOption::model_file)
      .def_readwrite("params_file", &RuntimeOption::params_file)
      .def_readwrite("model_format", &RuntimeOption::model_format)
      .def_readwrite("backend", &RuntimeOption::backend)
      .def_readwrite("external_stream", &RuntimeOption::external_stream_)
      .def_readwrite("model_from_memory", &RuntimeOption::model_from_memory_)
      .def_readwrite("cpu_thread_num", &RuntimeOption::cpu_thread_num)
      .def_readwrite("device_id", &RuntimeOption::device_id)
      .def_readwrite("device", &RuntimeOption::device);
}
} // namespace ultra_infer
