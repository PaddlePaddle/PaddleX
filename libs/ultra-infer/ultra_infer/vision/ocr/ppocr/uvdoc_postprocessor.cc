// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "ultra_infer/vision/ocr/ppocr/uvdoc_postprocessor.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

// bool UVDocPostprocessor::SingleBatchPostprocessor(const float* out_data,
// cv::Mat* result) {
//     // Reverse normalization
//     std::vector<float> mean{127.5f, 127.5f, 127.5f};
//     std::vector<float> std{127.5f, 127.5f, 127.5f};
//     Mat result_mat = Mat::Create(result->rows, result->cols, 3,
//     FDDataType::FP32, const_cast<float*>(out_data));
//     Convert::Run(&result_mat, mean, std);

//     // Convert result_mat to OpenCV Mat object
//     auto temp = result_mat.GetOpenCVMat();
//     cv::Mat res = cv::Mat::zeros(temp->size(), CV_8UC3);
//     temp->convertTo(res, CV_8UC3, 1);

//     // Execute BGR2RGB conversion
//     Mat fd_image = WrapMat(res);
//     BGR2RGB::Run(&fd_image);
//     res = *(fd_image.GetOpenCVMat());

//     // Copy result to output
//     res.copyTo(*result);

//     return true;
// }

bool UVDocPostprocessor::Run(const std::vector<FDTensor> &infer_results,
                             std::vector<FDTensor> *results) {
  *results = infer_results;
  return true;
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
