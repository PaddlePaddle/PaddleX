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

#include "ultra_infer/vision/sr/ppsr/ppmsvsr.h"

namespace ultra_infer {
namespace vision {
namespace sr {

PPMSVSR::PPMSVSR(const std::string &model_file, const std::string &params_file,
                 const RuntimeOption &custom_option,
                 const ModelFormat &model_format) {
  // unsupported ORT backend
  valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::OPENVINO};
  valid_gpu_backends = {Backend::PDINFER, Backend::TRT, Backend::ORT};

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;

  initialized = Initialize();
}

bool PPMSVSR::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  mean_ = {0., 0., 0.};
  scale_ = {1., 1., 1.};
  return true;
}

bool PPMSVSR::Preprocess(Mat *mat, std::vector<float> &output) {
  BGR2RGB::Run(mat);
  Normalize::Run(mat, mean_, scale_, true);
  HWC2CHW::Run(mat);
  // Csat float
  float *ptr = static_cast<float *>(mat->Data());
  size_t size = mat->Width() * mat->Height() * mat->Channels();
  output = std::vector<float>(ptr, ptr + size);
  return true;
}

bool PPMSVSR::Predict(std::vector<cv::Mat> &imgs,
                      std::vector<cv::Mat> &results) {
  // Theoretically, the more frame nums there are, the better the result will
  // be, but it will lead to a significant increase in memory
  int frame_num = imgs.size();
  int rows = imgs[0].rows;
  int cols = imgs[0].cols;
  int channels = imgs[0].channels();
  std::vector<FDTensor> input_tensors;
  input_tensors.resize(1);
  std::vector<float> all_data_temp;
  for (int i = 0; i < frame_num; i++) {
    Mat mat(imgs[i]);
    std::vector<float> data_temp;
    Preprocess(&mat, data_temp);
    all_data_temp.insert(all_data_temp.end(), data_temp.begin(),
                         data_temp.end());
  }
  // share memory in order to avoid memory copy, data type must be float32
  input_tensors[0].SetExternalData({1, frame_num, channels, rows, cols},
                                   FDDataType::FP32, all_data_temp.data());
  input_tensors[0].shape = {1, frame_num, channels, rows, cols};
  input_tensors[0].name = InputInfoOfRuntime(0).name;
  std::vector<FDTensor> output_tensors;
  if (!Infer(input_tensors, &output_tensors)) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }
  if (!Postprocess(output_tensors, results)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }
  return true;
}

bool PPMSVSR::Postprocess(std::vector<FDTensor> &infer_results,
                          std::vector<cv::Mat> &results) {
  // group to image
  // output_shape is [b, n, c, h, w] n = frame_nums b=1(default)
  // b and n is dependence export model shape
  // see
  // https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md
  auto output_shape = infer_results[0].shape;
  // PP-MSVSR
  int h_ = output_shape[3];
  int w_ = output_shape[4];
  int c_ = output_shape[2];
  int frame_num = output_shape[1];

  float *out_data = static_cast<float *>(infer_results[0].Data());
  cv::Mat temp = cv::Mat::zeros(h_, w_, CV_32FC3); // RGB image
  int pix_num = h_ * w_;
  int frame_pix_num = pix_num * c_;
  for (int frame = 0; frame < frame_num; frame++) {
    int index = 0;
    for (int h = 0; h < h_; ++h) {
      for (int w = 0; w < w_; ++w) {
        temp.at<cv::Vec3f>(h, w) = {
            out_data[2 * pix_num + index + frame_pix_num * frame],
            out_data[pix_num + index + frame_pix_num * frame],
            out_data[index + frame_pix_num * frame]};
        index += 1;
      }
    }
    // tmp data type is float[0-1.0],convert to uint type
    cv::Mat res = cv::Mat::zeros(temp.size(), CV_8UC3);
    temp.convertTo(res, CV_8UC3, 255);
    results.push_back(res);
  }
  return true;
}
} // namespace sr
} // namespace vision
} // namespace ultra_infer
