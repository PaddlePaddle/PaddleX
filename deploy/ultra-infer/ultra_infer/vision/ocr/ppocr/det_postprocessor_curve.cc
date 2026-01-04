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

#include "ultra_infer/vision/ocr/ppocr/det_postprocessor_curve.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

bool DBCURVEDetectorPostprocessor::SingleBatchPostprocessor(
    const float *out_data, int n2, int n3,
    const std::array<int, 4> &det_img_info,
    std::vector<std::vector<int>> *boxes_result) {
  int n = n2 * n3;

  // prepare bitmap
  std::vector<float> pred(n, 0.0);
  std::vector<unsigned char> cbuf(n, ' ');

  for (int i = 0; i < n; i++) {
    pred[i] = float(out_data[i]);
    cbuf[i] = (unsigned char)((out_data[i]) * 255);
  }
  cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
  cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

  const double threshold = det_db_thresh_ * 255;
  const double maxvalue = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
  if (use_dilation_) {
    cv::Mat dila_ele =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, bit_map, dila_ele);
  }

  std::vector<std::vector<std::vector<int>>> boxes;

  if (det_db_box_type_ == "bbox") {
    boxes = util_post_processor_.BoxesFromBitmap(
        pred_map, bit_map, det_db_box_thresh_, det_db_unclip_ratio_,
        det_db_score_mode_);
    boxes = util_post_processor_.FilterTagDetRes(boxes, det_img_info);
  } else {
    boxes = util_post_processor_.PolygonFromBitmap(
        pred_map, bit_map, det_db_box_thresh_, det_db_unclip_ratio_,
        det_db_score_mode_);
    boxes = util_post_processor_.FilterCURVETagDetRes(boxes, det_img_info);
  }

  // boxes to boxes_result
  for (int i = 0; i < boxes.size(); i++) {
    std::vector<int> new_box;
    for (auto &vec : boxes[i]) {
      for (auto &e : vec) {
        new_box.push_back(e);
      }
    }
    boxes_result->emplace_back(new_box);
  }

  return true;
}

bool DBCURVEDetectorPostprocessor::Run(
    const std::vector<FDTensor> &tensors,
    std::vector<std::vector<std::vector<int>>> *results,
    const std::vector<std::array<int, 4>> &batch_det_img_info) {
  // DBCURVEDetector have only 1 output tensor.
  const FDTensor &tensor = tensors[0];

  // For DBCURVEDetector, the output tensor shape = [batch, 1, ?, ?]
  size_t batch = tensor.shape[0];
  size_t length = accumulate(tensor.shape.begin() + 1, tensor.shape.end(), 1,
                             std::multiplies<int>());
  const float *tensor_data = reinterpret_cast<const float *>(tensor.Data());

  results->resize(batch);
  for (int i_batch = 0; i_batch < batch; ++i_batch) {
    SingleBatchPostprocessor(tensor_data, tensor.shape[2], tensor.shape[3],
                             batch_det_img_info[i_batch],
                             &results->at(i_batch));
    tensor_data = tensor_data + length;
  }
  return true;
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
