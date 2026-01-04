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

#include "adaptive_pool2d.h"

namespace ultra_infer {

nvinfer1::PluginFieldCollection AdaptivePool2dPluginCreator::mFC{};
std::vector<nvinfer1::PluginField>
    AdaptivePool2dPluginCreator::mPluginAttributes;

pluginStatus_t AdaptivePool2dInference(cudaStream_t stream, int32_t n,
                                       const void *input, void *output);

AdaptivePool2d::AdaptivePool2d(std::vector<int32_t> output_size,
                               std::string pooling_type) {
  output_size_ = output_size;
  pooling_type_ = pooling_type;
}

AdaptivePool2d::AdaptivePool2d(const void *buffer, size_t length) {
  const char *d = reinterpret_cast<const char *>(buffer), *a = d;
  output_size_.resize(4);
  for (int64_t i = 0; i < 4; i++) {
    output_size_[i] = read<int32_t>(d);
  }
  if (read<int32_t>(d) == 0) {
    pooling_type_ = "avg";
  } else {
    pooling_type_ = "max";
  }
  FDASSERT(d == a + length, "deserialize failed.");
}

int AdaptivePool2d::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs AdaptivePool2d::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  try {
    nvinfer1::DimsExprs output(inputs[0]);
    output.d[2] = exprBuilder.constant(static_cast<int32_t>(output_size_[2]));
    output.d[3] = exprBuilder.constant(static_cast<int32_t>(output_size_[3]));
    return output;
  } catch (const std::exception &e) {
    FDASSERT(false, "getOutputDimensions failed: %s.", e.what());
  }
  return nvinfer1::DimsExprs{};
}

int AdaptivePool2d::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                            const nvinfer1::PluginTensorDesc *outputDesc,
                            const void *const *inputs, void *const *outputs,
                            void *workspace, cudaStream_t stream) noexcept {
  int nums = outputDesc[0].dims.d[0] * outputDesc[0].dims.d[1] *
             outputDesc[0].dims.d[2] * outputDesc[0].dims.d[3];
  std::vector<int64_t> input_size, output_size;
  for (int i = 0; i < 4; i++) {
    input_size.push_back(inputDesc[0].dims.d[i]);
    output_size.push_back(outputDesc[0].dims.d[i]);
  }
  if (inputDesc[0].type == nvinfer1::DataType::kHALF) {
    if (outputDesc[0].type == nvinfer1::DataType::kHALF) {
      CudaAdaptivePool(input_size, output_size, outputs[0], inputs[0], stream,
                       pooling_type_, "half", "half");
    } else if (outputDesc[0].type == nvinfer1::DataType::kFLOAT) {
      CudaAdaptivePool(input_size, output_size, outputs[0], inputs[0], stream,
                       pooling_type_, "half", "float");
    }
  } else if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
    CudaAdaptivePool(input_size, output_size, outputs[0], inputs[0], stream,
                     pooling_type_, "float", "float");
  }
  return cudaPeekAtLastError();
}

size_t AdaptivePool2d::getSerializationSize() const noexcept {
  return 5 * sizeof(int32_t);
}

void AdaptivePool2d::serialize(void *buffer) const noexcept {
  char *d = reinterpret_cast<char *>(buffer), *a = d;
  for (int64_t i = 0; i < 4; i++) {
    write(d, output_size_[i]);
  }
  int32_t pooling_type_val = 0;
  if (pooling_type_ != "avg") {
    pooling_type_val = 1;
  }
  write(d, pooling_type_val);
  FDASSERT(d == a + getSerializationSize(), "d == a + getSerializationSize()");
}

nvinfer1::DataType
AdaptivePool2d::getOutputDataType(int index,
                                  const nvinfer1::DataType *inputType,
                                  int nbInputs) const noexcept {
  return inputType[0];
}

bool AdaptivePool2d::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) noexcept {
  if ((inOut[pos].format == nvinfer1::PluginFormat::kLINEAR) &&
      (inOut[pos].type == nvinfer1::DataType::kFLOAT ||
       inOut[pos].type == nvinfer1::DataType::kHALF)) {
    return true;
  }
  return false;
}

int AdaptivePool2d::initialize() noexcept { return 0; }

void AdaptivePool2d::terminate() noexcept { return; }

size_t AdaptivePool2d::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  return 0;
}

const char *AdaptivePool2d::getPluginType() const noexcept {
  return "AdaptivePool2d";
}

const char *AdaptivePool2d::getPluginVersion() const noexcept { return "1"; }

void AdaptivePool2d::destroy() noexcept { return; }
void AdaptivePool2d::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {
  return;
}
nvinfer1::IPluginV2DynamicExt *AdaptivePool2d::clone() const noexcept {
  try {
    nvinfer1::IPluginV2DynamicExt *plugin =
        new AdaptivePool2d(output_size_, pooling_type_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
  } catch (std::exception const &e) {
    FDASSERT(false, "clone failed: %s.", e.what());
  }
  return nullptr;
}

AdaptivePool2dPluginCreator::AdaptivePool2dPluginCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "output_size", nullptr, nvinfer1::PluginFieldType::kINT32, 4));
  mPluginAttributes.emplace_back(nvinfer1::PluginField(
      "pooling_type", nullptr, nvinfer1::PluginFieldType::kCHAR, 3));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *AdaptivePool2dPluginCreator::getPluginName() const noexcept {
  return "AdaptivePool2d";
}

const char *AdaptivePool2dPluginCreator::getPluginVersion() const noexcept {
  return "1";
}

const nvinfer1::PluginFieldCollection *
AdaptivePool2dPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

nvinfer1::IPluginV2DynamicExt *AdaptivePool2dPluginCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
    const nvinfer1::PluginField *fields = fc->fields;
    auto const dims = static_cast<int32_t const *>(fields[0].data);
    output_size_.resize(4);
    for (int64_t i = 0; i < 4; i++) {
      output_size_[i] = dims[i];
    }

    const char *pooling_type_ptr = (static_cast<char const *>(fields[1].data));
    std::string pooling_type(pooling_type_ptr, 3);
    pooling_type_ = pooling_type;
    return new AdaptivePool2d(output_size_, pooling_type_);
  } catch (std::exception const &e) {
    FDASSERT(false, "createPlugin failed: %s.", e.what());
  }
  return nullptr;
}

nvinfer1::IPluginV2DynamicExt *AdaptivePool2dPluginCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  try {
    return new AdaptivePool2d(serialData, serialLength);
  } catch (std::exception const &e) {
    FDASSERT(false, "deserializePlugin failed: %s.", e.what());
  }
  return nullptr;
}

} // namespace ultra_infer
