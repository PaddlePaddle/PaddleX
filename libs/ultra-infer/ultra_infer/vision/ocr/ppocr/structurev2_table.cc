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

#include "ultra_infer/vision/ocr/ppocr/structurev2_table.h"

#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

StructureV2Table::StructureV2Table() {}
StructureV2Table::StructureV2Table(const std::string &model_file,
                                   const std::string &params_file,
                                   const std::string &table_char_dict_path,
                                   const std::string &box_shape,
                                   const RuntimeOption &custom_option,
                                   const ModelFormat &model_format)
    : postprocessor_(table_char_dict_path, box_shape) {
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
bool StructureV2Table::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

std::unique_ptr<StructureV2Table> StructureV2Table::Clone() const {
  std::unique_ptr<StructureV2Table> clone_model =
      utils::make_unique<StructureV2Table>(StructureV2Table(*this));
  clone_model->SetRuntime(clone_model->CloneRuntime());
  return clone_model;
}

bool StructureV2Table::Predict(const cv::Mat &img,
                               std::vector<std::array<int, 8>> *boxes_result,
                               std::vector<std::string> *structure_result) {
  std::vector<std::vector<std::array<int, 8>>> det_results;
  std::vector<std::vector<std::string>> structure_results;
  if (!BatchPredict({img}, &det_results, &structure_results)) {
    return false;
  }
  *boxes_result = std::move(det_results[0]);
  *structure_result = std::move(structure_results[0]);
  return true;
}

bool StructureV2Table::Predict(const cv::Mat &img,
                               vision::OCRResult *ocr_result) {
  if (!Predict(img, &(ocr_result->table_boxes),
               &(ocr_result->table_structure))) {
    return false;
  }
  return true;
}

bool StructureV2Table::BatchPredict(
    const std::vector<cv::Mat> &images,
    std::vector<vision::OCRResult> *ocr_results) {
  std::vector<std::vector<std::array<int, 8>>> det_results;
  std::vector<std::vector<std::string>> structure_results;
  if (!BatchPredict(images, &det_results, &structure_results)) {
    return false;
  }
  ocr_results->resize(det_results.size());
  for (int i = 0; i < det_results.size(); i++) {
    (*ocr_results)[i].table_boxes = std::move(det_results[i]);
    (*ocr_results)[i].table_structure = std::move(structure_results[i]);
  }
  return true;
}

bool StructureV2Table::BatchPredict(
    const std::vector<cv::Mat> &images,
    std::vector<std::vector<std::array<int, 8>>> *det_results,
    std::vector<std::vector<std::string>> *structure_results) {
  std::vector<FDMat> fd_images = WrapMat(images);
  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }
  auto batch_det_img_info = preprocessor_.GetBatchImgInfo();

  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors_, det_results,
                          structure_results, *batch_det_img_info)) {
    FDERROR << "Failed to postprocess the inference cls_results by runtime."
            << std::endl;
    return false;
  }
  return true;
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
