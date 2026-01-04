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
#include "ultra_infer/vision/common/processors/mat.h"

#include "ultra_infer/utils/utils.h"
#include "ultra_infer/vision/common/processors/utils.h"

namespace ultra_infer {
namespace vision {

cv::Mat *Mat::GetOpenCVMat() {
  if (mat_type == ProcLib::OPENCV) {
    return &cpu_mat;
  } else if (mat_type == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    // Just a reference to fcv_mat, zero copy. After you
    // call this method, cpu_mat and fcv_mat will point
    // to the same memory buffer.
    cpu_mat = ConvertFlyCVMatToOpenCV(fcv_mat);
    mat_type = ProcLib::OPENCV;
    return &cpu_mat;
#else
    FDASSERT(false, "UltraInfer didn't compiled with FlyCV!");
#endif
  } else if (mat_type == ProcLib::CUDA || mat_type == ProcLib::CVCUDA) {
#ifdef WITH_GPU
    FDASSERT(cudaStreamSynchronize(stream) == cudaSuccess,
             "[ERROR] Error occurs while sync cuda stream.");
    cpu_mat = CreateZeroCopyOpenCVMatFromTensor(*fd_tensor, layout);
    mat_type = ProcLib::OPENCV;
    device = Device::CPU;
    return &cpu_mat;
#else
    FDASSERT(false, "UltraInfer didn't compiled with -DWITH_GPU=ON");
#endif
  } else {
    FDASSERT(false, "The mat_type of custom Mat can not be ProcLib::DEFAULT");
  }
}

#ifdef ENABLE_FLYCV
fcv::Mat *Mat::GetFlyCVMat() {
  if (mat_type == ProcLib::FLYCV) {
    return &fcv_mat;
  } else if (mat_type == ProcLib::OPENCV) {
    // Just a reference to cpu_mat, zero copy. After you
    // call this method, fcv_mat and cpu_mat will point
    // to the same memory buffer.
    fcv_mat = ConvertOpenCVMatToFlyCV(cpu_mat);
    mat_type = ProcLib::FLYCV;
    return &fcv_mat;
  } else {
    FDASSERT(false, "The mat_type of custom Mat can not be ProcLib::DEFAULT");
  }
}
#endif

void *Mat::Data() {
  if (mat_type == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    return fcv_mat.data();
#else
    FDASSERT(false,
             "UltraInfer didn't compile with FlyCV, but met data type with "
             "fcv::Mat.");
#endif
  } else if (device == Device::GPU) {
    return fd_tensor->Data();
  }
  return cpu_mat.ptr();
}

FDTensor *Mat::Tensor() {
  if (mat_type == ProcLib::OPENCV) {
    ShareWithTensor(fd_tensor.get());
  } else if (mat_type == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    cpu_mat = ConvertFlyCVMatToOpenCV(fcv_mat);
    mat_type = ProcLib::OPENCV;
    ShareWithTensor(fd_tensor.get());
#else
    FDASSERT(false, "UltraInfer didn't compiled with FlyCV!");
#endif
  }
  return fd_tensor.get();
}

void Mat::SetTensor(FDTensor *tensor) {
  fd_tensor->SetExternalData(tensor->Shape(), tensor->Dtype(), tensor->Data(),
                             tensor->device, tensor->device_id);
  device = tensor->device;
  if (layout == Layout::HWC) {
    height = tensor->Shape()[0];
    width = tensor->Shape()[1];
    channels = tensor->Shape()[2];
  } else if (layout == Layout::CHW) {
    channels = tensor->Shape()[0];
    height = tensor->Shape()[1];
    width = tensor->Shape()[2];
  }
}

void Mat::SetTensor(std::shared_ptr<FDTensor> &tensor) {
  fd_tensor = tensor;
  device = tensor->device;
  if (layout == Layout::HWC) {
    height = tensor->Shape()[0];
    width = tensor->Shape()[1];
    channels = tensor->Shape()[2];
  } else if (layout == Layout::CHW) {
    channels = tensor->Shape()[0];
    height = tensor->Shape()[1];
    width = tensor->Shape()[2];
  }
}

void Mat::ShareWithTensor(FDTensor *tensor) {
  tensor->SetExternalData({Channels(), Height(), Width()}, Type(), Data());
  tensor->device = device;
  if (layout == Layout::HWC) {
    tensor->shape = {Height(), Width(), Channels()};
  }
}

bool Mat::CopyToTensor(FDTensor *tensor) {
  int total_bytes = Height() * Width() * Channels() * FDDataTypeSize(Type());
  if (total_bytes != tensor->Nbytes()) {
    FDERROR << "While copy Mat to Tensor, requires the memory size be same, "
               "but now size of Tensor = "
            << tensor->Nbytes() << ", size of Mat = " << total_bytes << "."
            << std::endl;
    return false;
  }
  memcpy(tensor->MutableData(), Data(), total_bytes);
  return true;
}

void Mat::PrintInfo(const std::string &flag) {
  std::cout << flag << ": "
            << "DataType=" << Type() << ", "
            << "Channel=" << Channels() << ", "
            << "Height=" << Height() << ", "
            << "Width=" << Width() << ", "
            << "Mean=";
  if (mat_type == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    fcv::Scalar mean = fcv::mean(fcv_mat);
    for (int i = 0; i < Channels(); ++i) {
      std::cout << mean[i] << " ";
    }
    std::cout << std::endl;
#else
    FDASSERT(false,
             "UltraInfer didn't compile with FlyCV, but met data type with "
             "fcv::Mat.");
#endif
  } else if (mat_type == ProcLib::OPENCV) {
    cv::Scalar mean = cv::mean(cpu_mat);
    for (int i = 0; i < Channels(); ++i) {
      std::cout << mean[i] << " ";
    }
    std::cout << std::endl;
  } else if (mat_type == ProcLib::CUDA || mat_type == ProcLib::CVCUDA) {
#ifdef WITH_GPU
    FDASSERT(cudaStreamSynchronize(stream) == cudaSuccess,
             "[ERROR] Error occurs while sync cuda stream.");
    cv::Mat tmp_mat = CreateZeroCopyOpenCVMatFromTensor(*fd_tensor, layout);
    cv::Scalar mean = cv::mean(tmp_mat);
    for (int i = 0; i < Channels(); ++i) {
      std::cout << mean[i] << " ";
    }
    std::cout << std::endl;
#else
    FDASSERT(false, "UltraInfer didn't compiled with -DWITH_GPU=ON");
#endif
  }
}

FDDataType Mat::Type() {
  int type = -1;
  if (mat_type == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    return FlyCVDataTypeToFD(fcv_mat.type());
#else
    FDASSERT(false,
             "UltraInfer didn't compile with FlyCV, but met data type with "
             "fcv::Mat.");
#endif
  } else if (mat_type == ProcLib::CUDA || mat_type == ProcLib::CVCUDA) {
    return fd_tensor->Dtype();
  }
  return OpenCVDataTypeToFD(cpu_mat.type());
}

Mat Mat::Create(const FDTensor &tensor) {
  if (DefaultProcLib::default_lib == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    fcv::Mat tmp_fcv_mat = CreateZeroCopyFlyCVMatFromTensor(tensor);
    Mat mat = Mat(tmp_fcv_mat);
    return mat;
#else
    FDASSERT(false, "UltraInfer didn't compiled with FlyCV!");
#endif
  }
  cv::Mat tmp_ocv_mat = CreateZeroCopyOpenCVMatFromTensor(tensor);
  Mat mat = Mat(tmp_ocv_mat);
  return mat;
}

Mat Mat::Create(const FDTensor &tensor, ProcLib lib) {
  if (lib == ProcLib::DEFAULT) {
    return Create(tensor);
  }
  if (lib == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    fcv::Mat tmp_fcv_mat = CreateZeroCopyFlyCVMatFromTensor(tensor);
    Mat mat = Mat(tmp_fcv_mat);
    return mat;
#else
    FDASSERT(false, "UltraInfer didn't compiled with FlyCV!");
#endif
  }
  cv::Mat tmp_ocv_mat = CreateZeroCopyOpenCVMatFromTensor(tensor);
  Mat mat = Mat(tmp_ocv_mat);
  return mat;
}

Mat Mat::Create(int height, int width, int channels, FDDataType type,
                void *data) {
  if (DefaultProcLib::default_lib == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    fcv::Mat tmp_fcv_mat =
        CreateZeroCopyFlyCVMatFromBuffer(height, width, channels, type, data);
    Mat mat = Mat(tmp_fcv_mat);
    return mat;
#else
    FDASSERT(false, "UltraInfer didn't compiled with FlyCV!");
#endif
  }
  cv::Mat tmp_ocv_mat =
      CreateZeroCopyOpenCVMatFromBuffer(height, width, channels, type, data);
  Mat mat = Mat(tmp_ocv_mat);
  return mat;
}

Mat Mat::Create(int height, int width, int channels, FDDataType type,
                void *data, ProcLib lib) {
  if (lib == ProcLib::DEFAULT) {
    return Create(height, width, channels, type, data);
  }
  if (lib == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
    fcv::Mat tmp_fcv_mat =
        CreateZeroCopyFlyCVMatFromBuffer(height, width, channels, type, data);
    Mat mat = Mat(tmp_fcv_mat);
    return mat;
#else
    FDASSERT(false, "UltraInfer didn't compiled with FlyCV!");
#endif
  }
  cv::Mat tmp_ocv_mat =
      CreateZeroCopyOpenCVMatFromBuffer(height, width, channels, type, data);
  Mat mat = Mat(tmp_ocv_mat);
  return mat;
}

FDMat WrapMat(const cv::Mat &image) {
  FDMat mat(image);
  return mat;
}

std::vector<FDMat> WrapMat(const std::vector<cv::Mat> &images) {
  std::vector<FDMat> mats;
  for (size_t i = 0; i < images.size(); ++i) {
    mats.emplace_back(FDMat(images[i]));
  }
  return mats;
}

bool CheckShapeConsistency(std::vector<Mat> *mats) {
  if (mats == nullptr) {
    return true;
  }
  for (size_t i = 1; i < mats->size(); ++i) {
    if ((*mats)[i].Channels() != (*mats)[0].Channels() ||
        (*mats)[i].Width() != (*mats)[0].Width() ||
        (*mats)[i].Height() != (*mats)[0].Height()) {
      return false;
    }
  }
  return true;
}

FDTensor *CreateCachedGpuInputTensor(Mat *mat) {
#ifdef WITH_GPU
  FDTensor *src = mat->Tensor();
  // Need to make sure the tensor is pointed to the input_cache.
  if (src->Data() == mat->output_cache->Data()) {
    std::swap(mat->input_cache, mat->output_cache);
    std::swap(mat->input_cache->name, mat->output_cache->name);
  }
  if (src->device == Device::GPU) {
    return src;
  } else if (src->device == Device::CPU) {
    // Tensor on CPU, we need copy it from CPU to GPU
    FDASSERT(src->Shape().size() == 3, "The CPU tensor must has 3 dims.")
    mat->output_cache->Resize(src->Shape(), src->Dtype(), "output_cache",
                              Device::GPU);
    FDASSERT(cudaMemcpyAsync(mat->output_cache->Data(), src->Data(),
                             src->Nbytes(), cudaMemcpyHostToDevice,
                             mat->Stream()) == 0,
             "[ERROR] Error occurs while copy memory from CPU to GPU.");
    std::swap(mat->input_cache, mat->output_cache);
    std::swap(mat->input_cache->name, mat->output_cache->name);
    return mat->input_cache;
  } else {
    FDASSERT(false, "FDMat is on unsupported device: %d", src->device);
  }
#else
  FDASSERT(false, "UltraInfer didn't compile with WITH_GPU.");
#endif
  return nullptr;
}

} // namespace vision
} // namespace ultra_infer
