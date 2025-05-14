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

#include "ultra_infer/vision/ocr/ppocr/recognizer.h"

#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

Recognizer::Recognizer() {}

Recognizer::Recognizer(const std::string &model_file,
                       const std::string &params_file,
                       const std::string &label_path,
                       const RuntimeOption &custom_option,
                       const ModelFormat &model_format)
    : postprocessor_(label_path) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT, Backend::OPENVINO};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::OPENVINO,
                          Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    valid_sophgonpu_backends = {Backend::SOPHGOTPU};
    valid_rknpu_backends = {Backend::RKNPU2};
  }

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

// Init
bool Recognizer::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }

  return true;
}

std::unique_ptr<Recognizer> Recognizer::Clone() const {
  std::unique_ptr<Recognizer> clone_model =
      utils::make_unique<Recognizer>(Recognizer(*this));
  clone_model->SetRuntime(clone_model->CloneRuntime());
  return clone_model;
}

bool Recognizer::Predict(const cv::Mat &img, std::string *text,
                         float *rec_score) {
  std::vector<std::string> texts(1);
  std::vector<float> rec_scores(1);
  bool success = BatchPredict({img}, &texts, &rec_scores);
  if (!success) {
    return success;
  }
  *text = std::move(texts[0]);
  *rec_score = rec_scores[0];
  return true;
}

bool Recognizer::Predict(const cv::Mat &img, vision::OCRResult *ocr_result) {
  ocr_result->text.resize(1);
  ocr_result->rec_scores.resize(1);
  if (!Predict(img, &(ocr_result->text[0]), &(ocr_result->rec_scores[0]))) {
    return false;
  }
  return true;
}

bool Recognizer::BatchPredict(const std::vector<cv::Mat> &images,
                              std::vector<std::string> *texts,
                              std::vector<float> *rec_scores) {
  return BatchPredict(images, texts, rec_scores, 0, images.size(), {});
}

bool Recognizer::BatchPredict(const std::vector<cv::Mat> &images,
                              vision::OCRResult *ocr_result) {
  return BatchPredict(images, &(ocr_result->text), &(ocr_result->rec_scores));
}

bool Recognizer::BatchPredict(const std::vector<cv::Mat> &images,
                              std::vector<std::string> *texts,
                              std::vector<float> *rec_scores,
                              size_t start_index, size_t end_index,
                              const std::vector<int> &indices) {
  size_t total_size = images.size();
  if (indices.size() != 0 && indices.size() != total_size) {
    FDERROR << "indices.size() should be 0 or images.size()." << std::endl;
    return false;
  }
  std::vector<FDMat> fd_images = WrapMat(images);
  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_, start_index,
                         end_index, indices)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors_, texts, rec_scores,
                          start_index, total_size, indices)) {
    FDERROR << "Failed to postprocess the inference cls_results by runtime."
            << std::endl;
    return false;
  }
  return true;
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
