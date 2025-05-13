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

#include "ultra_infer/vision/sr/ppsr/edvr.h"

namespace ultra_infer {
namespace vision {
namespace sr {

EDVR::EDVR(const std::string &model_file, const std::string &params_file,
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

bool EDVR::Postprocess(std::vector<FDTensor> &infer_results,
                       std::vector<cv::Mat> &results) {
  // group to image
  // output_shape is [b, n, c, h, w] n = frame_nums b=1(default)
  // b and n is dependence export model shape
  // see
  // https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md
  auto output_shape = infer_results[0].shape;
  // EDVR
  int h_ = output_shape[2];
  int w_ = output_shape[3];
  int c_ = output_shape[1];
  int frame_num = 1;
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
