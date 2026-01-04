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

#include "ultra_infer/utils/utils.h"

#include "ultra_infer/vision/common/processors/utils.h"

namespace ultra_infer {
namespace vision {

FDDataType OpenCVDataTypeToFD(int type) {
  type = type % 8;
  if (type == 0) {
    return FDDataType::UINT8;
  } else if (type == 1) {
    return FDDataType::INT8;
  } else if (type == 2) {
    FDASSERT(false,
             "While calling OpenCVDataTypeToFD(), get UINT16 type which is not "
             "supported now.");
  } else if (type == 3) {
    return FDDataType::INT16;
  } else if (type == 4) {
    return FDDataType::INT32;
  } else if (type == 5) {
    return FDDataType::FP32;
  } else if (type == 6) {
    return FDDataType::FP64;
  } else {
    FDASSERT(false,
             "While calling OpenCVDataTypeToFD(), get type = %d, which is not "
             "expected.",
             type);
  }
}

int CreateOpenCVDataType(FDDataType type, int channel) {
  FDASSERT(channel == 1 || channel == 3 || channel == 4,
           "Only support channel be 1/3/4 in OpenCV.");
  if (type == FDDataType::UINT8) {
    if (channel == 1) {
      return CV_8UC1;
    } else if (channel == 3) {
      return CV_8UC3;
    } else {
      return CV_8UC4;
    }
  } else if (type == FDDataType::FP32) {
    if (channel == 1) {
      return CV_32FC1;
    } else if (channel == 3) {
      return CV_32FC3;
    } else {
      return CV_32FC4;
    }
  }
  FDASSERT(false, "Data type of %s is not supported.", Str(type).c_str());
  return CV_32FC3;
}

#ifdef ENABLE_FLYCV
FDDataType FlyCVDataTypeToFD(fcv::FCVImageType type) {
  if (type == fcv::FCVImageType::GRAY_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PKG_BGR_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PKG_RGB_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PKG_BGR_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PKG_RGB_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PLA_BGR_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PLA_RGB_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PLA_BGRA_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PLA_RGBA_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PLA_BGR_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PLA_RGB_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PLA_BGRA_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PLA_RGBA_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PKG_BGRA_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PKG_RGBA_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PKG_BGRA_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PKG_RGBA_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PKG_BGR565_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PKG_RGB565_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::GRAY_S32) {
    return FDDataType::INT32;
  } else if (type == fcv::FCVImageType::GRAY_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PKG_BGR_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PKG_RGB_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PKG_BGR_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PKG_RGB_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PKG_BGRA_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PKG_RGBA_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PKG_BGRA_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PKG_RGBA_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::GRAY_F64) {
    return FDDataType::FP64;
  }
  FDASSERT(false, "While calling FlyCVDataTypeToFD(), get unexpected type:%d.",
           int(type));
  return FDDataType::UNKNOWN1;
}

fcv::FCVImageType CreateFlyCVDataType(FDDataType type, int channel) {
  FDASSERT(channel == 1 || channel == 3 || channel == 4,
           "Only support channel be 1/3/4 in FlyCV.");
  if (type == FDDataType::UINT8) {
    if (channel == 1) {
      return fcv::FCVImageType::GRAY_U8;
    } else if (channel == 3) {
      return fcv::FCVImageType::PKG_BGR_U8;
    } else {
      return fcv::FCVImageType::PKG_BGRA_U8;
    }
  } else if (type == FDDataType::FP32) {
    if (channel == 1) {
      return fcv::FCVImageType::GRAY_F32;
    } else if (channel == 3) {
      return fcv::FCVImageType::PKG_BGR_F32;
    } else {
      return fcv::FCVImageType::PKG_BGRA_F32;
    }
  }
  FDASSERT(false, "Data type of %s is not supported.", Str(type).c_str());
  return fcv::FCVImageType::PKG_BGR_F32;
}

fcv::Mat ConvertOpenCVMatToFlyCV(cv::Mat &im) {
  int type = im.type() % 8;
  // 0: uint8; 5: float32; 6: float64
  if (type != 0 && type != 5 && type != 6) {
    FDASSERT(false, "Only support type of uint8/float/double, but now it's %d.",
             im.type());
  }
  auto fcv_type =
      CreateFlyCVDataType(OpenCVDataTypeToFD(im.type()), im.channels());
  return fcv::Mat(im.cols, im.rows, fcv_type, im.ptr()); // reference only
}

cv::Mat ConvertFlyCVMatToOpenCV(fcv::Mat &fim) {
  auto fd_dtype = FlyCVDataTypeToFD(fim.type());
  if (fd_dtype != FDDataType::UINT8 && fd_dtype != FDDataType::FP32 &&
      fd_dtype != FDDataType::FP64) {
    FDASSERT(false, "Only support type of uint8/float/double, but now it's %s.",
             Str(fd_dtype).c_str());
  }
  auto ocv_type = CreateOpenCVDataType(fd_dtype, fim.channels());
  return cv::Mat(fim.height(), fim.width(), ocv_type,
                 fim.data()); // reference only
}
#endif

cv::Mat CreateZeroCopyOpenCVMatFromBuffer(int height, int width, int channels,
                                          FDDataType type, void *data) {
  cv::Mat ocv_mat;
  switch (type) {
  case FDDataType::UINT8:
    ocv_mat = cv::Mat(height, width, CV_8UC(channels), data);
    break;
  case FDDataType::INT8:
    ocv_mat = cv::Mat(height, width, CV_8SC(channels), data);
    break;
  case FDDataType::INT16:
    ocv_mat = cv::Mat(height, width, CV_16SC(channels), data);
    break;
  case FDDataType::INT32:
    ocv_mat = cv::Mat(height, width, CV_32SC(channels), data);
    break;
  case FDDataType::FP32:
    ocv_mat = cv::Mat(height, width, CV_32FC(channels), data);
    break;
  case FDDataType::FP64:
    ocv_mat = cv::Mat(height, width, CV_64FC(channels), data);
    break;
  default:
    FDASSERT(false,
             "Tensor type %d is not supported While calling "
             "CreateZeroCopyOpenCVMat.",
             type);
    break;
  }
  return ocv_mat;
}

cv::Mat CreateZeroCopyOpenCVMatFromTensor(const FDTensor &tensor,
                                          Layout layout) {
  FDASSERT(tensor.shape.size() == 3, "When create OepnCV Mat from tensor,"
                                     "tensor shape should be 3-Dim");
  FDDataType type = tensor.dtype;
  int height = static_cast<int>(tensor.shape[0]);
  int width = static_cast<int>(tensor.shape[1]);
  int channels = static_cast<int>(tensor.shape[2]);
  if (layout == Layout::CHW) {
    channels = static_cast<int>(tensor.shape[0]);
    height = static_cast<int>(tensor.shape[1]);
    width = static_cast<int>(tensor.shape[2]);
  }
  return CreateZeroCopyOpenCVMatFromBuffer(
      height, width, channels, type, const_cast<void *>(tensor.CpuData()));
}

#ifdef ENABLE_FLYCV
fcv::Mat CreateZeroCopyFlyCVMatFromBuffer(int height, int width, int channels,
                                          FDDataType type, void *data) {
  fcv::Mat fcv_mat;
  auto fcv_type = CreateFlyCVDataType(type, channels);
  switch (type) {
  case FDDataType::UINT8:
    fcv_mat = fcv::Mat(width, height, fcv_type, data);
    break;
  case FDDataType::FP32:
    fcv_mat = fcv::Mat(width, height, fcv_type, data);
    break;
  case FDDataType::FP64:
    fcv_mat = fcv::Mat(width, height, fcv_type, data);
    break;
  default:
    FDASSERT(false,
             "Tensor type %d is not supported While calling "
             "CreateZeroCopyFlyCVMat.",
             type);
    break;
  }
  return fcv_mat;
}

fcv::Mat CreateZeroCopyFlyCVMatFromTensor(const FDTensor &tensor) {
  // TODO(qiuyanjun): Should add a Layout checking. Now, we
  // assume that the input tensor is already in Layout::HWC.
  FDASSERT(tensor.shape.size() == 3,
           "When create FlyCV Mat from tensor,"
           "tensor shape should be 3-Dim, HWC layout");
  FDDataType type = tensor.dtype;
  int height = static_cast<int>(tensor.shape[0]);
  int width = static_cast<int>(tensor.shape[1]);
  int channels = static_cast<int>(tensor.shape[2]);
  return CreateZeroCopyFlyCVMatFromBuffer(height, width, channels, type,
                                          const_cast<void *>(tensor.Data()));
}
#endif

} // namespace vision
} // namespace ultra_infer
