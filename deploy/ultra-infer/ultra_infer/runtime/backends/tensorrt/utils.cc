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

#include "ultra_infer/runtime/backends/tensorrt/utils.h"

namespace ultra_infer {

int ShapeRangeInfo::Update(const std::vector<int64_t> &new_shape) {
  if (new_shape.size() != shape.size()) {
    return -1;
  }
  int need_update_engine = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (is_static[i] == 1 && new_shape[i] != shape[i]) {
      return -1;
    }
    if (new_shape[i] < min[i] || min[i] < 0) {
      need_update_engine = 1;
    }
    if (new_shape[i] > max[i] || max[i] < 0) {
      need_update_engine = 1;
    }
  }

  if (need_update_engine == 0) {
    return 0;
  }

  FDWARNING << "[New Shape Out of Range] input name: " << name
            << ", shape: " << new_shape
            << ", The shape range before: min_shape=" << min
            << ", max_shape=" << max << "." << std::endl;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (new_shape[i] < min[i] || min[i] < 0) {
      min[i] = new_shape[i];
    }
    if (new_shape[i] > max[i] || max[i] < 0) {
      max[i] = new_shape[i];
    }
  }
  FDWARNING
      << "[New Shape Out of Range] The updated shape range now: min_shape="
      << min << ", max_shape=" << max << "." << std::endl;
  return need_update_engine;
}

size_t TrtDataTypeSize(const nvinfer1::DataType &dtype) {
  if (dtype == nvinfer1::DataType::kFLOAT) {
    return sizeof(float);
  } else if (dtype == nvinfer1::DataType::kHALF) {
    return sizeof(float) / 2;
  } else if (dtype == nvinfer1::DataType::kINT8) {
    return sizeof(int8_t);
  } else if (dtype == nvinfer1::DataType::kINT32) {
    return sizeof(int32_t);
  }
  // kBOOL
  return sizeof(bool);
}

FDDataType GetFDDataType(const nvinfer1::DataType &dtype) {
  if (dtype == nvinfer1::DataType::kFLOAT) {
    return FDDataType::FP32;
  } else if (dtype == nvinfer1::DataType::kHALF) {
    return FDDataType::FP16;
  } else if (dtype == nvinfer1::DataType::kINT8) {
    return FDDataType::INT8;
  } else if (dtype == nvinfer1::DataType::kINT32) {
    return FDDataType::INT32;
  }
  // kBOOL
  return FDDataType::BOOL;
}

nvinfer1::DataType ReaderDtypeToTrtDtype(int reader_dtype) {
  if (reader_dtype == 0) {
    return nvinfer1::DataType::kFLOAT;
  } else if (reader_dtype == 1) {
    FDASSERT(false, "TensorRT cannot support data type of double now.");
  } else if (reader_dtype == 2) {
    FDASSERT(false, "TensorRT cannot support data type of uint8 now.");
  } else if (reader_dtype == 3) {
    return nvinfer1::DataType::kINT8;
  } else if (reader_dtype == 4) {
    return nvinfer1::DataType::kINT32;
  } else if (reader_dtype == 5) {
    // regard int64 as int32
    return nvinfer1::DataType::kINT32;
  } else if (reader_dtype == 6) {
    return nvinfer1::DataType::kHALF;
  }
  FDASSERT(false, "Received unexpected data type of %d", reader_dtype);
  return nvinfer1::DataType::kFLOAT;
}

FDDataType ReaderDtypeToFDDtype(int reader_dtype) {
  if (reader_dtype == 0) {
    return FDDataType::FP32;
  } else if (reader_dtype == 1) {
    return FDDataType::FP64;
  } else if (reader_dtype == 2) {
    return FDDataType::UINT8;
  } else if (reader_dtype == 3) {
    return FDDataType::INT8;
  } else if (reader_dtype == 4) {
    return FDDataType::INT32;
  } else if (reader_dtype == 5) {
    return FDDataType::INT64;
  } else if (reader_dtype == 6) {
    return FDDataType::FP16;
  }
  FDASSERT(false, "Received unexpected data type of %d", reader_dtype);
  return FDDataType::FP32;
}

std::vector<int> ToVec(const nvinfer1::Dims &dim) {
  std::vector<int> out(dim.d, dim.d + dim.nbDims);
  return out;
}

int64_t Volume(const nvinfer1::Dims &d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

nvinfer1::Dims ToDims(const std::vector<int> &vec) {
  int limit = static_cast<int>(nvinfer1::Dims::MAX_DIMS);
  if (static_cast<int>(vec.size()) > limit) {
    FDWARNING << "Vector too long, only first 8 elements are used in dimension."
              << std::endl;
  }
  // Pick first nvinfer1::Dims::MAX_DIMS elements
  nvinfer1::Dims dims{std::min(static_cast<int>(vec.size()), limit), {}};
  std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
  return dims;
}

nvinfer1::Dims ToDims(const std::vector<int64_t> &vec) {
  int limit = static_cast<int>(nvinfer1::Dims::MAX_DIMS);
  if (static_cast<int>(vec.size()) > limit) {
    FDWARNING << "Vector too long, only first 8 elements are used in dimension."
              << std::endl;
  }
  // Pick first nvinfer1::Dims::MAX_DIMS elements
  nvinfer1::Dims dims{std::min(static_cast<int>(vec.size()), limit), {}};
  std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
  return dims;
}

} // namespace ultra_infer
