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
#pragma once
#include "opencv2/core/core.hpp"
#include "ultra_infer/core/fd_tensor.h"
#include "ultra_infer/vision/common/processors/proc_lib.h"

#ifdef ENABLE_FLYCV
#include "flycv.h" // NOLINT
#endif

#ifdef WITH_GPU
#include <cuda_runtime_api.h>
#endif

namespace ultra_infer {
namespace vision {

enum Layout { HWC, CHW };

/*! @brief FDMat is a structure for replace cv::Mat
 */
struct ULTRAINFER_DECL Mat {
  Mat() = default;
  explicit Mat(const cv::Mat &mat) {
    cpu_mat = mat;
    layout = Layout::HWC;
    height = cpu_mat.rows;
    width = cpu_mat.cols;
    channels = cpu_mat.channels();
    mat_type = ProcLib::OPENCV;
  }

#ifdef ENABLE_FLYCV
  explicit Mat(const fcv::Mat &mat) {
    fcv_mat = mat;
    layout = Layout::HWC;
    height = fcv_mat.height();
    width = fcv_mat.width();
    channels = fcv_mat.channels();
    mat_type = ProcLib::FLYCV;
  }
#endif

  Mat(const Mat &mat) = default;
  Mat &operator=(const Mat &mat) = default;

  // Move constructor
  Mat(Mat &&other) = default;

  // Careful if you use this interface
  // this only used if you don't want to write
  // the original data, and write to a new cv::Mat
  // then replace the old cv::Mat of this structure
  void SetMat(const cv::Mat &mat) {
    cpu_mat = mat;
    mat_type = ProcLib::OPENCV;
  }

  cv::Mat *GetOpenCVMat();

#ifdef ENABLE_FLYCV
  void SetMat(const fcv::Mat &mat) {
    fcv_mat = mat;
    mat_type = ProcLib::FLYCV;
  }
  fcv::Mat *GetFlyCVMat();
#endif

  void *Data();

  // Get fd_tensor
  FDTensor *Tensor();

  // Set fd_tensor
  void SetTensor(FDTensor *tensor);

  void SetTensor(std::shared_ptr<FDTensor> &tensor);

private:
  int channels;
  int height;
  int width;
  cv::Mat cpu_mat;
#ifdef ENABLE_FLYCV
  fcv::Mat fcv_mat;
#endif
#ifdef WITH_GPU
  cudaStream_t stream = nullptr;
#endif
  // Currently, fd_tensor is only used by CUDA and CV-CUDA,
  // OpenCV and FlyCV are not using it.
  std::shared_ptr<FDTensor> fd_tensor = std::make_shared<FDTensor>();

public:
  FDDataType Type();
  int Channels() const { return channels; }
  int Width() const { return width; }
  int Height() const { return height; }
  void SetChannels(int s) { channels = s; }
  void SetWidth(int w) { width = w; }
  void SetHeight(int h) { height = h; }

  // When using CV-CUDA/CUDA, please set input/output cache,
  // refer to manager.cc
  FDTensor *input_cache = nullptr;
  FDTensor *output_cache = nullptr;
#ifdef WITH_GPU
  cudaStream_t Stream() const { return stream; }
  void SetStream(cudaStream_t s) { stream = s; }
#endif

  // Transfer the vision::Mat to FDTensor
  void ShareWithTensor(FDTensor *tensor);
  // Only support copy to cpu tensor now
  bool CopyToTensor(FDTensor *tensor);

  // Debug functions
  // TODO(jiangjiajun) Develop a right process pipeline with c++
  // is not a easy things, Will add more debug function here to
  // help debug processed image. This function will print shape
  // and mean of each channels of the Mat
  void PrintInfo(const std::string &flag);

  ProcLib mat_type = ProcLib::OPENCV;
  Layout layout = Layout::HWC;
  Device device = Device::CPU;
  ProcLib proc_lib = ProcLib::DEFAULT;

  // Create FD Mat from FD Tensor. This method only create a
  // new FD Mat with zero copy and it's data pointer is reference
  // to the original memory buffer of input FD Tensor. Carefully,
  // any operation on this Mat may change memory that points to
  // FDTensor. We assume that the memory Mat points to is mutable.
  // This method will create a FD Mat according to current global
  // default ProcLib (OPENCV,FLYCV,...).
  static Mat Create(const FDTensor &tensor);
  static Mat Create(const FDTensor &tensor, ProcLib lib);
  static Mat Create(int height, int width, int channels, FDDataType type,
                    void *data);
  static Mat Create(int height, int width, int channels, FDDataType type,
                    void *data, ProcLib lib);
};

typedef Mat FDMat;
/*
 * @brief Wrap a cv::Mat to FDMat, there's no memory copy, memory buffer is
 * managed by user
 */
ULTRAINFER_DECL FDMat WrapMat(const cv::Mat &image);
/*
 * Warp a vector<cv::Mat> to vector<FDMat>, there's no memory copy, memory
 * buffer is managed by user
 */
ULTRAINFER_DECL std::vector<FDMat> WrapMat(const std::vector<cv::Mat> &images);

bool CheckShapeConsistency(std::vector<Mat> *mats);

// Create an input tensor on GPU and save into input_cache.
// If the Mat is on GPU, return the mat->Tensor() directly.
// If the Mat is on CPU, then update the input cache tensor and copy the mat's
// CPU tensor to this new GPU input cache tensor.
FDTensor *CreateCachedGpuInputTensor(Mat *mat);
} // namespace vision
} // namespace ultra_infer
