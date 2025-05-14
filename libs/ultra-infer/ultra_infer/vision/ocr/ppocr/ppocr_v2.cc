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

#include "ultra_infer/vision/ocr/ppocr/ppocr_v2.h"
#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace pipeline {
PPOCRv2::PPOCRv2(ultra_infer::vision::ocr::DBDetector *det_model,
                 ultra_infer::vision::ocr::Classifier *cls_model,
                 ultra_infer::vision::ocr::Recognizer *rec_model)
    : detector_(det_model), classifier_(cls_model), recognizer_(rec_model) {
  Initialized();
  auto preprocess_shape = recognizer_->GetPreprocessor().GetRecImageShape();
  preprocess_shape[1] = 32;
  recognizer_->GetPreprocessor().SetRecImageShape(preprocess_shape);
}

PPOCRv2::PPOCRv2(ultra_infer::vision::ocr::DBDetector *det_model,
                 ultra_infer::vision::ocr::Recognizer *rec_model)
    : detector_(det_model), recognizer_(rec_model) {
  Initialized();
  auto preprocess_shape = recognizer_->GetPreprocessor().GetRecImageShape();
  preprocess_shape[1] = 32;
  recognizer_->GetPreprocessor().SetRecImageShape(preprocess_shape);
}

bool PPOCRv2::SetClsBatchSize(int cls_batch_size) {
  if (cls_batch_size < -1 || cls_batch_size == 0) {
    FDERROR << "batch_size > 0 or batch_size == -1." << std::endl;
    return false;
  }
  cls_batch_size_ = cls_batch_size;
  return true;
}

int PPOCRv2::GetClsBatchSize() { return cls_batch_size_; }

bool PPOCRv2::SetRecBatchSize(int rec_batch_size) {
  if (rec_batch_size < -1 || rec_batch_size == 0) {
    FDERROR << "batch_size > 0 or batch_size == -1." << std::endl;
    return false;
  }
  rec_batch_size_ = rec_batch_size;
  return true;
}

int PPOCRv2::GetRecBatchSize() { return rec_batch_size_; }

bool PPOCRv2::Initialized() const {

  if (detector_ != nullptr && !detector_->Initialized()) {
    return false;
  }

  if (classifier_ != nullptr && !classifier_->Initialized()) {
    return false;
  }

  if (recognizer_ != nullptr && !recognizer_->Initialized()) {
    return false;
  }
  return true;
}

std::unique_ptr<PPOCRv2> PPOCRv2::Clone() const {
  std::unique_ptr<PPOCRv2> clone_model =
      utils::make_unique<PPOCRv2>(PPOCRv2(*this));
  clone_model->detector_ = detector_->Clone().release();
  if (classifier_ != nullptr) {
    clone_model->classifier_ = classifier_->Clone().release();
  }
  clone_model->recognizer_ = recognizer_->Clone().release();
  return clone_model;
}

bool PPOCRv2::Predict(cv::Mat *img, ultra_infer::vision::OCRResult *result) {
  return Predict(*img, result);
}

bool PPOCRv2::Predict(const cv::Mat &img,
                      ultra_infer::vision::OCRResult *result) {
  std::vector<ultra_infer::vision::OCRResult> batch_result(1);
  bool success = BatchPredict({img}, &batch_result);
  if (!success) {
    return success;
  }
  *result = std::move(batch_result[0]);
  return true;
};

bool PPOCRv2::BatchPredict(
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

    if (nullptr != classifier_) {
      for (size_t start_index = 0; start_index < image_list.size();
           start_index += cls_batch_size_) {
        size_t end_index =
            std::min(start_index + cls_batch_size_, image_list.size());
        if (!classifier_->BatchPredict(image_list, cls_labels_ptr,
                                       cls_scores_ptr, start_index,
                                       end_index)) {
          FDERROR << "There's error while recognizing image in PPOCR."
                  << std::endl;
          return false;
        } else {
          for (size_t i_img = start_index; i_img < end_index; ++i_img) {
            if (cls_labels_ptr->at(i_img) % 2 == 1 &&
                cls_scores_ptr->at(i_img) >
                    classifier_->GetPostprocessor().GetClsThresh()) {
              cv::rotate(image_list[i_img], image_list[i_img], 1);
            }
          }
        }
      }
    }

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
  return true;
}

} // namespace pipeline
} // namespace ultra_infer
