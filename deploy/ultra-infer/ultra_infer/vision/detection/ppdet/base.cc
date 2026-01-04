#include "ultra_infer/vision/detection/ppdet/base.h"

#include "ultra_infer/utils/unique_ptr.h"
#include "ultra_infer/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"

namespace ultra_infer {
namespace vision {
namespace detection {

PPDetBase::PPDetBase(const std::string &model_file,
                     const std::string &params_file,
                     const std::string &config_file,
                     const RuntimeOption &custom_option,
                     const ModelFormat &model_format)
    : preprocessor_(config_file), postprocessor_(preprocessor_.GetArch()) {
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
}

std::unique_ptr<PPDetBase> PPDetBase::Clone() const {
  std::unique_ptr<PPDetBase> clone_model =
      ultra_infer::utils::make_unique<PPDetBase>(PPDetBase(*this));
  clone_model->SetRuntime(clone_model->CloneRuntime());
  return clone_model;
}

bool PPDetBase::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  return true;
}

bool PPDetBase::Predict(cv::Mat *im, DetectionResult *result) {
  return Predict(*im, result);
}

bool PPDetBase::Predict(const cv::Mat &im, DetectionResult *result) {
  std::vector<DetectionResult> results;
  if (!BatchPredict({im}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool PPDetBase::BatchPredict(const std::vector<cv::Mat> &imgs,
                             std::vector<DetectionResult> *results) {
  std::vector<FDMat> fd_images = WrapMat(imgs);
  if (!preprocessor_.Run(&fd_images, &reused_input_tensors_)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }
  reused_input_tensors_[0].name = "image";
  reused_input_tensors_[1].name = "scale_factor";
  reused_input_tensors_[2].name = "im_shape";

  if (NumInputsOfRuntime() == 1) {
    auto scale_factor = static_cast<float *>(reused_input_tensors_[1].Data());
    postprocessor_.SetScaleFactor({scale_factor[0], scale_factor[1]});
  }

  // Some models don't need scale_factor and im_shape as input
  while (reused_input_tensors_.size() != NumInputsOfRuntime()) {
    reused_input_tensors_.pop_back();
  }

  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }

  if (!postprocessor_.Run(reused_output_tensors_, results)) {
    FDERROR << "Failed to postprocess the inference results by runtime."
            << std::endl;
    return false;
  }
  return true;
}

bool PPDetBase::CheckArch() {
  // Add "PicoDet" arch for backward compatibility with the
  // old ppdet model, such as picodet from PaddleClas
  // PP-ShiTuV2 pipeline.
  std::vector<std::string> archs = {
      "SOLOv2", "YOLO",   "SSD",    "RetinaNet", "RCNN",   "Face",
      "GFL",    "YOLOX",  "YOLOv5", "YOLOv6",    "YOLOv7", "RTMDet",
      "FCOS",   "TTFNet", "TOOD",   "DETR",      "PicoDet"};
  auto arch_ = preprocessor_.GetArch();
  for (auto item : archs) {
    if (arch_ == item) {
      return true;
    }
  }
  FDWARNING << "Please set model arch,"
            << "support value : SOLOv2, YOLO, SSD, RetinaNet, "
            << "RCNN, Face , GFL , RTMDet ,"
            << "FCOS , TTFNet , TOOD , DETR, PicoDet" << std::endl;
  return false;
}

} // namespace detection
} // namespace vision
} // namespace ultra_infer
