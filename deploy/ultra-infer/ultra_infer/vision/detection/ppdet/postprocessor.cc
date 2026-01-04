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

#include "ultra_infer/vision/detection/ppdet/postprocessor.h"

#include "ultra_infer/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"

namespace ultra_infer {
namespace vision {
namespace detection {

bool PaddleDetPostprocessor::ProcessMask(
    const FDTensor &tensor, std::vector<DetectionResult> *results) {
  auto shape = tensor.Shape();
  int64_t out_mask_w = shape[2];
  int64_t out_mask_numel = shape[1] * shape[2];
  const auto *data = reinterpret_cast<const uint32_t *>(tensor.CpuData());
  int index = 0;

  for (int i = 0; i < results->size(); ++i) {
    (*results)[i].contain_masks = true;
    (*results)[i].masks.resize((*results)[i].boxes.size());
    for (int j = 0; j < (*results)[i].boxes.size(); ++j) {
      int x1 = static_cast<int>(round((*results)[i].boxes[j][0]));
      int y1 = static_cast<int>(round((*results)[i].boxes[j][1]));
      int x2 = static_cast<int>(round((*results)[i].boxes[j][2]));
      int y2 = static_cast<int>(round((*results)[i].boxes[j][3]));
      int keep_mask_h = y2 - y1;
      int keep_mask_w = x2 - x1;
      int keep_mask_numel = keep_mask_h * keep_mask_w;
      (*results)[i].masks[j].Resize(keep_mask_numel);
      (*results)[i].masks[j].shape = {keep_mask_h, keep_mask_w};
      const uint32_t *current_ptr = data + index * out_mask_numel;

      auto *keep_mask_ptr =
          reinterpret_cast<uint32_t *>((*results)[i].masks[j].Data());
      for (int row = y1; row < y2; ++row) {
        size_t keep_nbytes_in_col = keep_mask_w * sizeof(uint32_t);
        const uint32_t *out_row_start_ptr = current_ptr + row * out_mask_w + x1;
        uint32_t *keep_row_start_ptr = keep_mask_ptr + (row - y1) * keep_mask_w;
        std::memcpy(keep_row_start_ptr, out_row_start_ptr, keep_nbytes_in_col);
      }
      index += 1;
    }
  }
  return true;
}

bool PaddleDetPostprocessor::ProcessWithNMS(
    const std::vector<FDTensor> &tensors,
    std::vector<DetectionResult> *results) {
  // Get number of boxes for each input image
  std::vector<int> num_boxes(tensors[1].shape[0]);
  int total_num_boxes = 0;
  if (tensors[1].dtype == FDDataType::INT32) {
    const auto *data = static_cast<const int32_t *>(tensors[1].CpuData());
    for (size_t i = 0; i < tensors[1].shape[0]; ++i) {
      num_boxes[i] = static_cast<int>(data[i]);
      total_num_boxes += num_boxes[i];
    }
  } else if (tensors[1].dtype == FDDataType::INT64) {
    const auto *data = static_cast<const int64_t *>(tensors[1].CpuData());
    for (size_t i = 0; i < tensors[1].shape[0]; ++i) {
      num_boxes[i] = static_cast<int>(data[i]);
      total_num_boxes += num_boxes[i];
    }
  }

  // Special case for TensorRT, it has fixed output shape of NMS
  // So there's invalid boxes in its' output boxes
  int num_output_boxes = static_cast<int>(tensors[0].Shape()[0]);
  bool contain_invalid_boxes = false;
  if (total_num_boxes != num_output_boxes) {
    if (num_output_boxes % num_boxes.size() == 0) {
      contain_invalid_boxes = true;
    } else {
      FDERROR << "Cannot handle the output data for this model, unexpected "
                 "situation."
              << std::endl;
      return false;
    }
  }

  // Get boxes for each input image
  results->resize(num_boxes.size());

  if (tensors[0].shape[0] == 0) {
    // No detected boxes
    return true;
  }

  const auto *box_data = static_cast<const float *>(tensors[0].CpuData());
  int offset = 0;
  for (size_t i = 0; i < num_boxes.size(); ++i) {
    const float *ptr = box_data + offset;
    (*results)[i].Reserve(num_boxes[i]);
    for (size_t j = 0; j < num_boxes[i]; ++j) {
      (*results)[i].label_ids.push_back(
          static_cast<int32_t>(round(ptr[j * 6])));
      (*results)[i].scores.push_back(ptr[j * 6 + 1]);
      (*results)[i].boxes.emplace_back(std::array<float, 4>(
          {ptr[j * 6 + 2], ptr[j * 6 + 3], ptr[j * 6 + 4], ptr[j * 6 + 5]}));
    }
    if (contain_invalid_boxes) {
      offset += static_cast<int>(num_output_boxes * 6 / num_boxes.size());
    } else {
      offset += static_cast<int>(num_boxes[i] * 6);
    }
  }
  return true;
}

bool PaddleDetPostprocessor::ProcessWithoutNMS(
    const std::vector<FDTensor> &tensors,
    std::vector<DetectionResult> *results) {
  int boxes_index = 0;
  int scores_index = 1;

  // Judge the index of the input Tensor
  if (tensors[0].shape[1] == tensors[1].shape[2]) {
    boxes_index = 0;
    scores_index = 1;
  } else if (tensors[0].shape[2] == tensors[1].shape[1]) {
    boxes_index = 1;
    scores_index = 0;
  } else {
    FDERROR << "The shape of boxes and scores should be [batch, boxes_num, "
               "4], [batch, classes_num, boxes_num]"
            << std::endl;
    return false;
  }

  // do multi class nms
  multi_class_nms_.Compute(
      static_cast<const float *>(tensors[boxes_index].Data()),
      static_cast<const float *>(tensors[scores_index].Data()),
      tensors[boxes_index].shape, tensors[scores_index].shape);
  auto num_boxes = multi_class_nms_.out_num_rois_data;
  auto box_data =
      static_cast<const float *>(multi_class_nms_.out_box_data.data());

  // Get boxes for each input image
  results->resize(num_boxes.size());
  int offset = 0;
  for (size_t i = 0; i < num_boxes.size(); ++i) {
    const float *ptr = box_data + offset;
    (*results)[i].Reserve(num_boxes[i]);
    for (size_t j = 0; j < num_boxes[i]; ++j) {
      (*results)[i].label_ids.push_back(
          static_cast<int32_t>(round(ptr[j * 6])));
      (*results)[i].scores.push_back(ptr[j * 6 + 1]);
      (*results)[i].boxes.emplace_back(std::array<float, 4>(
          {ptr[j * 6 + 2], ptr[j * 6 + 3], ptr[j * 6 + 4], ptr[j * 6 + 5]}));
    }
    offset += (num_boxes[i] * 6);
  }

  // do scale
  if (GetScaleFactor()[0] != 0) {
    for (auto &result : *results) {
      for (auto &box : result.boxes) {
        box[0] /= GetScaleFactor()[1];
        box[1] /= GetScaleFactor()[0];
        box[2] /= GetScaleFactor()[1];
        box[3] /= GetScaleFactor()[0];
      }
    }
  }
  return true;
}

bool PaddleDetPostprocessor::ProcessSolov2(
    const std::vector<FDTensor> &tensors,
    std::vector<DetectionResult> *results) {
  if (tensors.size() != 4) {
    FDERROR << "The size of tensors for solov2 must be 4." << std::endl;
    return false;
  }

  if (tensors[0].shape[0] != 1) {
    FDERROR << "SOLOv2 temporarily only supports batch size is 1." << std::endl;
    return false;
  }

  results->clear();
  results->resize(1);

  (*results)[0].contain_masks = true;

  // tensor[0] means bbox data
  const auto bbox_data = static_cast<const int *>(tensors[0].CpuData());
  // tensor[1] means label data
  const auto label_data_ = static_cast<const int64_t *>(tensors[1].CpuData());
  // tensor[2] means score data
  const auto score_data_ = static_cast<const float *>(tensors[2].CpuData());
  // tensor[3] is mask data and its shape is the same as that of the image.
  const auto mask_data_ = static_cast<const uint8_t *>(tensors[3].CpuData());

  int rows = static_cast<int>(tensors[3].shape[1]);
  int cols = static_cast<int>(tensors[3].shape[2]);
  for (int bbox_id = 0; bbox_id < bbox_data[0]; ++bbox_id) {
    if (score_data_[bbox_id] >= multi_class_nms_.score_threshold) {
      DetectionResult &result_item = (*results)[0];
      result_item.label_ids.emplace_back(label_data_[bbox_id]);
      result_item.scores.emplace_back(score_data_[bbox_id]);

      std::vector<int> global_mask;

      for (int k = 0; k < rows * cols; ++k) {
        global_mask.push_back(
            static_cast<int>(mask_data_[k + bbox_id * rows * cols]));
      }

      // find minimize bounding box from mask
      cv::Mat mask(rows, cols, CV_32SC1);

      std::memcpy(mask.data, global_mask.data(),
                  global_mask.size() * sizeof(int));

      cv::Mat mask_fp;
      mask.convertTo(mask_fp, CV_32FC1);

      cv::Mat rowSum;
      cv::Mat colSum;
      std::vector<float> sum_of_row(rows);
      std::vector<float> sum_of_col(cols);
      cv::reduce(mask_fp, colSum, 0, cv::REDUCE_SUM, CV_32FC1);
      cv::reduce(mask_fp, rowSum, 1, cv::REDUCE_SUM, CV_32FC1);

      for (int row_id = 0; row_id < rows; ++row_id) {
        sum_of_row[row_id] = rowSum.at<float>(row_id, 0);
      }
      for (int col_id = 0; col_id < cols; ++col_id) {
        sum_of_col[col_id] = colSum.at<float>(0, col_id);
      }

      auto it = std::find_if(sum_of_row.begin(), sum_of_row.end(),
                             [](int x) { return x > 0.5; });
      float y1 = std::distance(sum_of_row.begin(), it);
      auto it2 = std::find_if(sum_of_col.begin(), sum_of_col.end(),
                              [](int x) { return x > 0.5; });
      float x1 = std::distance(sum_of_col.begin(), it2);
      auto rit = std::find_if(sum_of_row.rbegin(), sum_of_row.rend(),
                              [](int x) { return x > 0.5; });
      float y2 = std::distance(rit, sum_of_row.rend());
      auto rit2 = std::find_if(sum_of_col.rbegin(), sum_of_col.rend(),
                               [](int x) { return x > 0.5; });
      float x2 = std::distance(rit2, sum_of_col.rend());
      result_item.boxes.emplace_back(std::array<float, 4>({x1, y1, x2, y2}));
    }
  }
  return true;
}

bool PaddleDetPostprocessor::ProcessPPYOLOER(
    const std::vector<FDTensor> &tensors,
    std::vector<DetectionResult> *results) {
  if (tensors.size() != 2) {
    FDERROR << "The size of tensors for PPYOLOER must be 2." << std::endl;
    return false;
  }

  int boxes_index = 0;
  int scores_index = 1;
  multi_class_nms_rotated_.Compute(
      static_cast<const float *>(tensors[boxes_index].Data()),
      static_cast<const float *>(tensors[scores_index].Data()),
      tensors[boxes_index].shape, tensors[scores_index].shape);
  auto num_boxes = multi_class_nms_rotated_.out_num_rois_data;
  auto box_data =
      static_cast<const float *>(multi_class_nms_rotated_.out_box_data.data());

  // Get boxes for each input image
  results->resize(num_boxes.size());
  int offset = 0;
  for (size_t i = 0; i < num_boxes.size(); ++i) {
    const float *ptr = box_data + offset;
    (*results)[i].Reserve(num_boxes[i]);
    for (size_t j = 0; j < num_boxes[i]; ++j) {
      (*results)[i].label_ids.push_back(
          static_cast<int32_t>(round(ptr[j * 10])));
      (*results)[i].scores.push_back(ptr[j * 10 + 1]);
      (*results)[i].rotated_boxes.push_back(std::array<float, 8>(
          {ptr[j * 10 + 2], ptr[j * 10 + 3], ptr[j * 10 + 4], ptr[j * 10 + 5],
           ptr[j * 10 + 6], ptr[j * 10 + 7], ptr[j * 10 + 8],
           ptr[j * 10 + 9]}));
    }
    offset += (num_boxes[i] * 10);
  }

  // do scale
  if (GetScaleFactor()[0] != 0) {
    for (auto &result : *results) {
      for (int i = 0; i < result.rotated_boxes.size(); i++) {
        for (int j = 0; j < 8; j++) {
          auto scale = i % 2 == 0 ? GetScaleFactor()[1] : GetScaleFactor()[0];
          result.rotated_boxes[i][j] /= float(scale);
        }
      }
    }
  }

  return true;
}

bool PaddleDetPostprocessor::Run(const std::vector<FDTensor> &tensors,
                                 std::vector<DetectionResult> *results) {
  if (arch_ == "SOLOv2") {
    // process for SOLOv2
    ProcessSolov2(tensors, results);
    // The fourth output of solov2 is mask
    return ProcessMask(tensors[3], results);
  } else {
    if (tensors[0].Shape().size() == 3 &&
        tensors[0].Shape()[2] == 8) { // PPYOLOER
      return ProcessPPYOLOER(tensors, results);
    }

    // Do process according to whether NMS exists.
    if (with_nms_) {
      if (!ProcessWithNMS(tensors, results)) {
        return false;
      }
    } else {
      if (!ProcessWithoutNMS(tensors, results)) {
        return false;
      }
    }

    // for only detection
    if (tensors.size() <= 2) {
      return true;
    }

    // for maskrcnn
    if (tensors[2].Shape()[0] != tensors[0].Shape()[0]) {
      FDERROR << "The first dimension of output mask tensor:"
              << tensors[2].Shape()[0]
              << " is not equal to the first dimension of output boxes tensor:"
              << tensors[0].Shape()[0] << "." << std::endl;
      return false;
    }

    // The third output of mask-rcnn is mask
    return ProcessMask(tensors[2], results);
  }
}
} // namespace detection
} // namespace vision
} // namespace ultra_infer
