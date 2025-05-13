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

#include "ultra_infer/vision/generation/contrib/postprocessor.h"

namespace ultra_infer {
namespace vision {
namespace generation {

bool AnimeGANPostprocessor::Run(std::vector<FDTensor> &infer_results,
                                std::vector<cv::Mat> *results) {
  // 1. Reverse normalization
  // 2. RGB2BGR
  FDTensor &output_tensor = infer_results.at(0);
  std::vector<int64_t> shape = output_tensor.Shape(); // n, h, w, c
  int size = shape[1] * shape[2] * shape[3];
  results->resize(shape[0]);
  float *infer_result_data = reinterpret_cast<float *>(output_tensor.Data());
  for (size_t i = 0; i < results->size(); ++i) {
    Mat result_mat = Mat::Create(shape[1], shape[2], 3, FDDataType::FP32,
                                 infer_result_data + i * size);
    std::vector<float> mean{127.5f, 127.5f, 127.5f};
    std::vector<float> std{127.5f, 127.5f, 127.5f};
    Convert::Run(&result_mat, mean, std);
    // tmp data type is float[0-1.0],convert to uint type
    auto temp = result_mat.GetOpenCVMat();
    cv::Mat res = cv::Mat::zeros(temp->size(), CV_8UC3);
    temp->convertTo(res, CV_8UC3, 1);
    Mat fd_image = WrapMat(res);
    BGR2RGB::Run(&fd_image);
    res = *(fd_image.GetOpenCVMat());
    res.copyTo(results->at(i));
  }
  return true;
}

} // namespace generation
} // namespace vision
} // namespace ultra_infer
