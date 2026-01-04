#include "ultra_infer/runtime/backends/tvm/tvm_backend.h"

#include "yaml-cpp/yaml.h"
namespace ultra_infer {
bool TVMBackend::Init(const ultra_infer::RuntimeOption &runtime_option) {
  if (!(Supported(runtime_option.model_format, Backend::TVM) &&
        Supported(runtime_option.device, Backend::TVM))) {
    FDERROR << "TVMBackend only supports model "
               "ModelFormat::TVMFormat/Backend::TVM, but now its "
            << runtime_option.model_format << "/" << runtime_option.device
            << std::endl;
    return false;
  }

  if (runtime_option.model_from_memory_) {
    FDERROR << "TVMBackend doesn't support load model from memory, please "
               "load model from disk."
            << std::endl;
    return false;
  }

  if (!BuildDLDevice(runtime_option.device)) {
    FDERROR << "TVMBackend only don't support run in this device." << std::endl;
    return false;
  }

  if (!BuildModel(runtime_option)) {
    FDERROR << "TVMBackend only don't support run with this model path."
            << std::endl;
    return false;
  }

  if (!InitInputAndOutputTensor()) {
    FDERROR << "InitInputAndOutputTensor failed." << std::endl;
    return false;
  }
  return true;
}

bool TVMBackend::InitInputAndOutputTensor() {
  input_tensor_.resize(NumInputs());
  for (int i = 0; i < NumInputs(); ++i) {
    TensorInfo tensor_info = GetInputInfo(i);
    tvm::ShapeTuple shape(tensor_info.shape.begin(), tensor_info.shape.end());
    input_tensor_[i] = tvm::runtime::NDArray::Empty(
        shape, FDDataTypeToDLDataType(tensor_info.dtype), dev_);
  }

  output_tensor_.resize(NumOutputs());
  for (int i = 0; i < NumOutputs(); ++i) {
    TensorInfo tensor_info = GetOutputInfo(i);
    tvm::ShapeTuple shape(tensor_info.shape.begin(), tensor_info.shape.end());
    output_tensor_[i] = tvm::runtime::NDArray::Empty(
        shape, FDDataTypeToDLDataType(tensor_info.dtype), dev_);
  }
  return true;
}

bool TVMBackend::BuildModel(const RuntimeOption &runtime_option) {
  // load in the library
  tvm::runtime::Module mod_factory =
      tvm::runtime::Module::LoadFromFile(runtime_option.model_file);

  // create the graph executor module
  gmod_ = mod_factory.GetFunction("default")(dev_);

  // load params
  std::ifstream params_in(runtime_option.params_file, std::ios::binary);
  std::string params_data((std::istreambuf_iterator<char>(params_in)),
                          std::istreambuf_iterator<char>());
  params_in.close();
  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();
  tvm::runtime::PackedFunc load_params = gmod_.GetFunction("load_params");
  load_params(params_arr);

  // read input and output info
  tvm::runtime::PackedFunc get_input_info = gmod_.GetFunction("get_input_info");
  tvm::Map<tvm::String, tvm::ObjectRef> input_info = get_input_info();
  auto input_info_shape = tvm::Downcast<tvm::Map<tvm::String, tvm::ShapeTuple>>(
      input_info["shape"]);
  inputs_desc_.reserve(input_info_shape.size());
  for (auto map_node : input_info_shape) {
    std::string temp_name = map_node.first;

    tvm::ShapeTuple tup = map_node.second;
    std::vector<int> temp_shape{};
    temp_shape.resize(tup.size());
    for (int j = 0; j < tup.size(); ++j) {
      temp_shape[j] = static_cast<int>(tup[j]);
    }

    FDDataType temp_dtype = ultra_infer::UNKNOWN1;
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    inputs_desc_.emplace_back(temp_input_info);
  }

  int input_dtype_index = 0;
  auto input_info_dtype =
      tvm::Downcast<tvm::Map<tvm::String, tvm::String>>(input_info["dtype"]);
  for (auto map_node : input_info_dtype) {
    tvm::String tup = map_node.second;
    inputs_desc_[input_dtype_index].dtype = TVMTensorTypeToFDDataType(tup);
    input_dtype_index++;
  }

  tvm::runtime::PackedFunc get_output_info =
      gmod_.GetFunction("get_output_info");
  tvm::Map<tvm::String, tvm::ObjectRef> output_info = get_output_info();
  auto output_info_shape =
      tvm::Downcast<tvm::Map<tvm::String, tvm::ShapeTuple>>(
          output_info["shape"]);
  outputs_desc_.reserve(output_info_shape.size());
  for (auto map_node : output_info_shape) {
    std::string temp_name = map_node.first;

    tvm::ShapeTuple tup = map_node.second;
    std::vector<int> temp_shape{};
    temp_shape.resize(tup.size());
    for (int j = 0; j < tup.size(); ++j) {
      temp_shape[j] = static_cast<int>(tup[j]);
    }

    FDDataType temp_dtype = ultra_infer::FP32;
    TensorInfo temp_input_info = {temp_name, temp_shape, temp_dtype};
    outputs_desc_.emplace_back(temp_input_info);
  }

  int output_dtype_index = 0;
  auto output_info_dtype =
      tvm::Downcast<tvm::Map<tvm::String, tvm::String>>(output_info["dtype"]);
  for (auto map_node : output_info_dtype) {
    tvm::String tup = map_node.second;
    outputs_desc_[output_dtype_index].dtype = TVMTensorTypeToFDDataType(tup);
    output_dtype_index++;
  }
  return true;
}

FDDataType TVMBackend::TVMTensorTypeToFDDataType(tvm::String type) {
  if (type == "float32") {
    return FDDataType::FP32;
  }
  FDERROR << "FDDataType don't support this type" << std::endl;
  return FDDataType::UNKNOWN1;
}

bool TVMBackend::Infer(std::vector<FDTensor> &inputs,
                       std::vector<FDTensor> *outputs, bool copy_to_fd) {
  for (int i = 0; i < inputs.size(); ++i) {
    memcpy(input_tensor_[i]->data, inputs[i].Data(), inputs[i].Nbytes());
  }

  // get the function from the module(set input data)
  tvm::runtime::PackedFunc set_input = gmod_.GetFunction("set_input");
  for (int i = 0; i < NumInputs(); ++i) {
    set_input(GetInputInfo(i).name, input_tensor_[i]);
  }

  // get the function from the module(run it)
  tvm::runtime::PackedFunc run = gmod_.GetFunction("run");
  run();

  // get the function from the module(get output data)
  tvm::runtime::PackedFunc get_output = gmod_.GetFunction("get_output");
  for (int i = 0; i < NumOutputs(); ++i) {
    get_output(i, output_tensor_[i]);
  }

  // get result
  outputs->resize(NumOutputs());
  std::vector<int64_t> temp_shape{};
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    temp_shape.resize(outputs_desc_[i].shape.size());
    for (int j = 0; j < outputs_desc_[i].shape.size(); ++j) {
      temp_shape[j] = outputs_desc_[i].shape[j];
    }
    (*outputs)[i].Resize(temp_shape, outputs_desc_[i].dtype,
                         outputs_desc_[i].name);
    memcpy((*outputs)[i].MutableData(),
           static_cast<float *>(output_tensor_[i]->data),
           (*outputs)[i].Nbytes());
  }
  return true;
}

bool TVMBackend::BuildDLDevice(ultra_infer::Device device) {
  if (device == Device::CPU) {
    dev_ = DLDevice{kDLCPU, 0};
  } else {
    FDERROR << "TVMBackend only support run in CPU." << std::endl;
    return false;
  }
  return true;
}

DLDataType TVMBackend::FDDataTypeToDLDataType(ultra_infer::FDDataType dtype) {
  if (dtype == FDDataType::FP32) {
    return DLDataType{kDLFloat, 32, 1};
  }
  return {};
}
} // namespace ultra_infer
