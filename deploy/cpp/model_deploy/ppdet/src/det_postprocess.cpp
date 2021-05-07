// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "model_deploy/ppdet/include/det_postprocess.h"

namespace PaddleDeploy {

bool DetPostprocess::Init(const YAML::Node& yaml_config) {
  labels_.clear();
  for (auto item : yaml_config["labels"]) {
    std::string label = item.as<std::string>();
    labels_.push_back(label);
  }
  version_ = yaml_config["version"].as<std::string>();
  return true;
}

bool DetPostprocess::ProcessBbox(const std::vector<DataBlob>& outputs,
                                 const std::vector<ShapeInfo>& shape_infos,
                                 std::vector<Result>* results, int thread_num) {
  const float* data = reinterpret_cast<const float*>(outputs[0].data.data());

  std::vector<int> num_bboxes_each_sample;
  if (outputs[0].lod.empty()) {
    num_bboxes_each_sample.push_back(outputs[0].data.size() / sizeof(float));
  } else {
    for (auto i = 0; i < outputs[0].lod[0].size() - 1; ++i) {
      int num = outputs[0].lod[0][i + 1] - outputs[0].lod[0][i];
      num_bboxes_each_sample.push_back(num);
    }
  }

  int idx = 0;
  for (auto i = 0; i < num_bboxes_each_sample.size(); ++i) {
    (*results)[i].model_type = "det";
    (*results)[i].det_result = new DetResult();
    for (auto j = 0; j < num_bboxes_each_sample[i]; ++j) {
      Box box;
      box.category_id = static_cast<int>(round(data[idx * 6]));
      if (box.category_id < 0) {
        std::cerr << "Compute category id is less than 0"
                  << "(Maybe no object detected)" << std::endl;
        return true;
      }
      if (box.category_id >= labels_.size()) {
        std::cerr << "Compute category id is greater than labels "
                  << "in your config file" << std::endl;
        std::cerr << "Compute Category ID: " << box.category_id
                  << ", but length of labels is " << labels_.size()
                  << std::endl;
        return false;
      }
      box.category = labels_[box.category_id];
      box.score = data[idx * 6 + 1];
      // TODO(jiangjiajun): only for RCNN and YOLO
      // lack of process for SSD and Face
      float xmin = data[idx * 6 + 2];
      float ymin = data[idx * 6 + 3];
      float xmax = data[idx * 6 + 4];
      float ymax = data[idx * 6 + 5];
      box.coordinate = {xmin, ymin, xmax - xmin, ymax - ymin};
      (*results)[i].det_result->boxes.push_back(std::move(box));
      idx += 1;
    }
  }
  return true;
}

bool DetPostprocess::ProcessMask(DataBlob* mask_blob,
                                 const std::vector<ShapeInfo>& shape_infos,
                                 std::vector<Result>* results, int thread_num) {
  std::vector<int> output_mask_shape = mask_blob->shape;
  float *mask_data = reinterpret_cast<float*>(mask_blob->data.data());
  int mask_pixels = output_mask_shape[2] * output_mask_shape[3];
  int classes = output_mask_shape[1];
  auto begin_mask_data = mask_data;
  for (int i = 0; i < results->size(); ++i) {
    (*results)[i].det_result->mask_resolution = output_mask_shape[2];
    for (int j = 0; j < (*results)[i].det_result->boxes.size(); ++j) {
      Box *box = &(*results)[i].det_result->boxes[j];
      int category_id = box->category_id;
      auto begin_mask = begin_mask_data + box->category_id * mask_pixels;
      cv::Mat bin_mask(output_mask_shape[2],
                      output_mask_shape[3],
                      CV_32FC1,
                      begin_mask);
      cv::resize(bin_mask, bin_mask, 
                 cv::Size(box->coordinate[2], box->coordinate[3]));
      
      cv::threshold(bin_mask, bin_mask, 0.5, 1, cv::THRESH_BINARY);
      bin_mask.convertTo(bin_mask, CV_8UC1);
      int max_w = shape_infos[i].shapes[0][0];
      int max_h = shape_infos[i].shapes[0][1];
      int padding_top = max_h - box->coordinate[1] - box->coordinate[3];
      int padding_bottom = box->coordinate[1];
      int padding_left = box->coordinate[0];
      int padding_right = max_w - box->coordinate[0] - box->coordinate[2];
      cv::Scalar value = cv::Scalar(0.0);
      cv::copyMakeBorder(bin_mask, bin_mask,
                         padding_top,
                         padding_bottom,
                         padding_left,
                         padding_right,
                         cv::BORDER_CONSTANT,
                         value=value)
      box->mask.shape = {max_w, max_h};
      auto mask_int_begin = reinterpret_cast<u_int8_t*>(bin_mask.data);
      auto mask_int_end =
        mask_int_begin + box->mask.shape[0] * box->mask.shape[1];
      box->mask.data.assign(mask_int_begin, mask_int_end);
      begin_mask_data += classes * mask_pixels;
    }
  }
  return true;
}

bool DetPostprocess::ProcessMaskV2(DataBlob* mask_blob,
                                 const std::vector<ShapeInfo>& shape_infos,
                                 std::vector<Result>* results, int thread_num) {
  std::vector<int> output_mask_shape = mask_blob->shape;
  float *mask_data = reinterpret_cast<float*>(mask_blob->data.data());
  int mask_pixels = output_mask_shape[1] * output_mask_shape[2];
  for (int i = 0; i < results->size(); ++i) {
    for (int j = 0; j < (*results)[i].det_result->boxes.size(); ++j) {
      Box *box = &(*results)[i].det_result->boxes[j];
      box->mask.shape = {static_cast<int>(output_mask_shape[1]),
                      static_cast<int>(output_mask_shape[2])};
      auto begin_mask = mask_data + j * mask_pixels;
      cv::Mat bin_mask(output_mask_shape[1],
                       output_mask_shape[2],
                       CV_32SC1,
                       begin_mask);
      bin_mask.convertTo(bin_mask, CV_8UC1);
      auto mask_int_begin = reinterpret_cast<u_int8_t*>(bin_mask.data);
      auto mask_int_end =
        mask_int_begin + box->mask.shape[0] * box->mask.shape[1];
      box->mask.data.assign(mask_int_begin, mask_int_end);
    }
  }
  return true;
}

bool DetPostprocess::Run(const std::vector<DataBlob>& outputs,
                         const std::vector<ShapeInfo>& shape_infos,
                         std::vector<Result>* results, int thread_num) {
  results->clear();
  if (outputs.size() == 0) {
    std::cerr << "empty input image on DetPostprocess" << std::endl;
    return true;
  }
  results->resize(shape_infos.size());
  if (!ProcessBbox(outputs, shape_infos, results, thread_num)) {
    std::cerr << "Error happend while process bboxes" << std::endl;
    return false;
  }
  // TODO(jiangjiajun): MaskRCNN is not implement
  if (version_ < "2.0" && outputs.size() == 2) {
    DataBlob mask_blob = outputs[1];
    if (!ProcessMask(&mask_blob, shape_infos, results, thread_num)) {
      std::cerr << "Error happend while process masks" << std::endl;
      return false;
    }
  } else if (version_ >= "2.0" && outputs.size() == 3) {
    DataBlob mask_blob = outputs[2];
    if (!ProcessMaskV2(&mask_blob, shape_infos, results, thread_num)) {
      std::cerr << "Error happend while process masks" << std::endl;
      return false;
    }
  }
  return true;
}

}  //  namespace PaddleDeploy
