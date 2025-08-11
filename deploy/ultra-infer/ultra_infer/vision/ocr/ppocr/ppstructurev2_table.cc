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

#include "ultra_infer/vision/ocr/ppocr/ppstructurev2_table.h"

#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace pipeline {
PPStructureV2Table::PPStructureV2Table(
    ultra_infer::vision::ocr::DBDetector *det_model,
    ultra_infer::vision::ocr::Recognizer *rec_model,
    ultra_infer::vision::ocr::StructureV2Table *table_model)
    : detector_(det_model), recognizer_(rec_model), table_(table_model) {
  Initialized();
}

bool PPStructureV2Table::SetRecBatchSize(int rec_batch_size) {
  if (rec_batch_size < -1 || rec_batch_size == 0) {
    FDERROR << "batch_size > 0 or batch_size == -1." << std::endl;
    return false;
  }
  rec_batch_size_ = rec_batch_size;
  return true;
}

int PPStructureV2Table::GetRecBatchSize() { return rec_batch_size_; }

bool PPStructureV2Table::Initialized() const {
  if (detector_ != nullptr && !detector_->Initialized()) {
    return false;
  }

  if (recognizer_ != nullptr && !recognizer_->Initialized()) {
    return false;
  }

  if (table_ != nullptr && !table_->Initialized()) {
    return false;
  }
  return true;
}

std::unique_ptr<PPStructureV2Table> PPStructureV2Table::Clone() const {
  std::unique_ptr<PPStructureV2Table> clone_model =
      utils::make_unique<PPStructureV2Table>(PPStructureV2Table(*this));
  clone_model->detector_ = detector_->Clone().release();
  clone_model->recognizer_ = recognizer_->Clone().release();
  clone_model->table_ = table_->Clone().release();
  return clone_model;
}

bool PPStructureV2Table::Predict(cv::Mat *img,
                                 ultra_infer::vision::OCRResult *result) {
  return Predict(*img, result);
}

bool PPStructureV2Table::Predict(const cv::Mat &img,
                                 ultra_infer::vision::OCRResult *result) {
  std::vector<ultra_infer::vision::OCRResult> batch_result(1);
  bool success = BatchPredict({img}, &batch_result);
  if (!success) {
    return success;
  }
  *result = std::move(batch_result[0]);
  return true;
};

bool PPStructureV2Table::BatchPredict(
    const std::vector<cv::Mat> &images,
    std::vector<ultra_infer::vision::OCRResult> *batch_result) {
  batch_result->clear();
  batch_result->resize(images.size());
  std::vector<std::vector<std::array<int, 8>>> batch_boxes(images.size());

  if (!detector_->BatchPredict(images, &batch_boxes)) {
    FDERROR << "There's error while detecting image in PPOCR." << std::endl;
    return false;
  }

  for (int i_batch = 0; i_batch < batch_boxes.size(); ++i_batch) {
    vision::ocr::SortBoxes(&(batch_boxes[i_batch]));
    (*batch_result)[i_batch].boxes = batch_boxes[i_batch];
  }

  for (int i_batch = 0; i_batch < images.size(); ++i_batch) {
    ultra_infer::vision::OCRResult &ocr_result = (*batch_result)[i_batch];
    // Get croped images by detection result
    const std::vector<std::array<int, 8>> &boxes = ocr_result.boxes;
    const cv::Mat &img = images[i_batch];
    std::vector<cv::Mat> image_list;
    if (boxes.size() == 0) {
      image_list.emplace_back(img);
    } else {
      image_list.resize(boxes.size());
      for (size_t i_box = 0; i_box < boxes.size(); ++i_box) {
        image_list[i_box] = vision::ocr::GetRotateCropImage(img, boxes[i_box]);
      }
    }
    std::vector<int32_t> *cls_labels_ptr = &ocr_result.cls_labels;
    std::vector<float> *cls_scores_ptr = &ocr_result.cls_scores;

    std::vector<std::string> *text_ptr = &ocr_result.text;
    std::vector<float> *rec_scores_ptr = &ocr_result.rec_scores;

    std::vector<float> width_list;
    for (int i = 0; i < image_list.size(); i++) {
      width_list.push_back(float(image_list[i].cols) / image_list[i].rows);
    }
    std::vector<int> indices = vision::ocr::ArgSort(width_list);

    for (size_t start_index = 0; start_index < image_list.size();
         start_index += rec_batch_size_) {
      size_t end_index =
          std::min(start_index + rec_batch_size_, image_list.size());
      if (!recognizer_->BatchPredict(image_list, text_ptr, rec_scores_ptr,
                                     start_index, end_index, indices)) {
        FDERROR << "There's error while recognizing image in PPOCR."
                << std::endl;
        return false;
      }
    }
  }

  if (!table_->BatchPredict(images, batch_result)) {
    FDERROR << "There's error while recognizing tables in images." << std::endl;
    return false;
  }

  for (int i_batch = 0; i_batch < batch_boxes.size(); ++i_batch) {
    ultra_infer::vision::OCRResult &ocr_result = (*batch_result)[i_batch];
    std::vector<std::vector<std::string>> matched(ocr_result.table_boxes.size(),
                                                  std::vector<std::string>());

    std::vector<int> ocr_box;
    std::vector<int> structure_box;
    for (int i = 0; i < ocr_result.boxes.size(); i++) {
      ocr_box = vision::ocr::Xyxyxyxy2Xyxy(ocr_result.boxes[i]);
      ocr_box[0] -= 1;
      ocr_box[1] -= 1;
      ocr_box[2] += 1;
      ocr_box[3] += 1;

      std::vector<std::vector<float>> dis_list(ocr_result.table_boxes.size(),
                                               std::vector<float>(3, 100000.0));

      for (int j = 0; j < ocr_result.table_boxes.size(); j++) {
        structure_box = vision::ocr::Xyxyxyxy2Xyxy(ocr_result.table_boxes[j]);
        dis_list[j][0] = vision::ocr::Dis(ocr_box, structure_box);
        dis_list[j][1] = 1 - vision::ocr::Iou(ocr_box, structure_box);
        dis_list[j][2] = j;
      }

      // find min dis idx
      std::sort(dis_list.begin(), dis_list.end(), vision::ocr::ComparisonDis);
      matched[dis_list[0][2]].push_back(ocr_result.text[i]);
    }

    // get pred html
    std::string html_str = "";
    int td_tag_idx = 0;
    auto structure_html_tags = ocr_result.table_structure;
    for (int i = 0; i < structure_html_tags.size(); i++) {
      if (structure_html_tags[i].find("</td>") != std::string::npos) {
        if (structure_html_tags[i].find("<td></td>") != std::string::npos) {
          html_str += "<td>";
        }
        if (matched[td_tag_idx].size() > 0) {
          bool b_with = false;
          if (matched[td_tag_idx][0].find("<b>") != std::string::npos &&
              matched[td_tag_idx].size() > 1) {
            b_with = true;
            html_str += "<b>";
          }
          for (int j = 0; j < matched[td_tag_idx].size(); j++) {
            std::string content = matched[td_tag_idx][j];
            if (matched[td_tag_idx].size() > 1) {
              // remove blank, <b> and </b>
              if (content.length() > 0 && content.at(0) == ' ') {
                content = content.substr(0);
              }
              if (content.length() > 2 && content.substr(0, 3) == "<b>") {
                content = content.substr(3);
              }
              if (content.length() > 4 &&
                  content.substr(content.length() - 4) == "</b>") {
                content = content.substr(0, content.length() - 4);
              }
              if (content.empty()) {
                continue;
              }
              // add blank
              if (j != matched[td_tag_idx].size() - 1 &&
                  content.at(content.length() - 1) != ' ') {
                content += ' ';
              }
            }
            html_str += content;
          }
          if (b_with) {
            html_str += "</b>";
          }
        }
        if (structure_html_tags[i].find("<td></td>") != std::string::npos) {
          html_str += "</td>";
        } else {
          html_str += structure_html_tags[i];
        }
        td_tag_idx += 1;
      } else {
        html_str += structure_html_tags[i];
      }
    }
    (*batch_result)[i_batch].table_html = html_str;
  }

  return true;
}

} // namespace pipeline
} // namespace ultra_infer
