#include "ultra_infer/vision/keypointdet/pptinypose/pptinypose.h"

#include "ultra_infer/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"
#ifdef ENABLE_PADDLE2ONNX
#include "paddle2onnx/converter.h"
#endif
#include "ultra_infer/vision.h"

namespace ultra_infer {
namespace vision {
namespace keypointdetection {

PPTinyPose::PPTinyPose(const std::string &model_file,
                       const std::string &params_file,
                       const std::string &config_file,
                       const RuntimeOption &custom_option,
                       const ModelFormat &model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::OPENVINO,
                        Backend::LITE};
  valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  valid_kunlunxin_backends = {Backend::LITE};
  valid_rknpu_backends = {Backend::RKNPU2};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool PPTinyPose::Initialize() {
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
    return false;
  }
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }

  return true;
}

bool PPTinyPose::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile &e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  std::string arch = cfg["arch"].as<std::string>();
  if (arch != "HRNet" && arch != "HigherHRNet") {
    FDERROR << "Require the arch of model is HRNet or HigherHRNet, but arch "
            << "defined in "
            << "config file is " << arch << "." << std::endl;
    return false;
  }

  processors_.push_back(std::make_shared<BGR2RGB>());

  for (const auto &op : cfg["Preprocess"]) {
    std::string op_name = op["type"].as<std::string>();
    if (op_name == "NormalizeImage") {
      if (!disable_normalize_) {
        auto mean = op["mean"].as<std::vector<float>>();
        auto std = op["std"].as<std::vector<float>>();
        bool is_scale = op["is_scale"].as<bool>();
        processors_.push_back(std::make_shared<Normalize>(mean, std, is_scale));
      }
    } else if (op_name == "Permute") {
      if (!disable_permute_) {
        // permute = cast<float> + HWC2CHW
        processors_.push_back(std::make_shared<Cast>("float"));
        processors_.push_back(std::make_shared<HWC2CHW>());
      }
    } else if (op_name == "TopDownEvalAffine") {
      auto trainsize = op["trainsize"].as<std::vector<int>>();
      int height = trainsize[1];
      int width = trainsize[0];
      cv::Mat trans_matrix(2, 3, CV_64FC1);
      processors_.push_back(
          std::make_shared<WarpAffine>(trans_matrix, width, height, 1));
    } else {
      FDERROR << "Unexcepted preprocess operator: " << op_name << "."
              << std::endl;
      return false;
    }
  }
  return true;
}

bool PPTinyPose::Preprocess(Mat *mat, std::vector<FDTensor> *outputs) {
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (processors_[i]->Name().compare("WarpAffine") == 0) {
      auto processor = dynamic_cast<WarpAffine *>(processors_[i].get());
      float origin_width = static_cast<float>(mat->Width());
      float origin_height = static_cast<float>(mat->Height());
      std::vector<float> center = {origin_width / 2.0f, origin_height / 2.0f};
      std::vector<float> scale = {origin_width, origin_height};
      int resize_width = -1;
      int resize_height = -1;
      std::tie(resize_width, resize_height) = processor->GetWidthAndHeight();
      cv::Mat trans_matrix(2, 3, CV_64FC1);
      GetAffineTransform(center, scale, 0, {resize_width, resize_height},
                         &trans_matrix, 0);
      if (!(processor->SetTransformMatrix(trans_matrix))) {
        FDERROR << "Failed to set transform matrix of "
                << processors_[i]->Name() << " processor." << std::endl;
      }
    }
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }

  outputs->resize(1);
  (*outputs)[0].name = InputInfoOfRuntime(0).name;
  mat->ShareWithTensor(&((*outputs)[0]));

  // reshape to [1, c, h, w]
  (*outputs)[0].ExpandDim(0);

  return true;
}

bool PPTinyPose::Postprocess(std::vector<FDTensor> &infer_result,
                             KeyPointDetectionResult *result,
                             const std::vector<float> &center,
                             const std::vector<float> &scale) {
  FDASSERT(infer_result[0].shape[0] == 1,
           "Only support batch = 1 in UltraInfer now.");
  result->Clear();

  if (infer_result.size() == 1) {
    FDTensor result_copy = infer_result[0];
    result_copy.Reshape({result_copy.shape[0], result_copy.shape[1],
                         result_copy.shape[2] * result_copy.shape[3]});
    infer_result.resize(2);
    function::ArgMax(result_copy, &infer_result[1], -1);
  }

  // Calculate output length
  int outdata_size =
      std::accumulate(infer_result[0].shape.begin(),
                      infer_result[0].shape.end(), 1, std::multiplies<int>());
  int idxdata_size =
      std::accumulate(infer_result[1].shape.begin(),
                      infer_result[1].shape.end(), 1, std::multiplies<int>());

  if (outdata_size < 6) {
    FDWARNING << "PPTinyPose No object detected." << std::endl;
  }
  float *out_data = static_cast<float *>(infer_result[0].Data());
  void *idx_data = infer_result[1].Data();
  int idx_dtype = infer_result[1].dtype;
  std::vector<int> out_data_shape(infer_result[0].shape.begin(),
                                  infer_result[0].shape.end());
  std::vector<int> idx_data_shape(infer_result[1].shape.begin(),
                                  infer_result[1].shape.end());
  std::vector<float> preds(out_data_shape[1] * 3, 0);
  std::vector<float> heatmap(out_data, out_data + outdata_size);
  std::vector<int64_t> idxout(idxdata_size);
  if (idx_dtype == FDDataType::INT32) {
    std::copy(static_cast<int32_t *>(idx_data),
              static_cast<int32_t *>(idx_data) + idxdata_size, idxout.begin());
  } else if (idx_dtype == FDDataType::INT64) {
    std::copy(static_cast<int64_t *>(idx_data),
              static_cast<int64_t *>(idx_data) + idxdata_size, idxout.begin());
  } else {
    FDERROR << "Only support process inference result with INT32/INT64 data "
               "type, but now it's "
            << idx_dtype << "." << std::endl;
  }
  GetFinalPredictions(heatmap, out_data_shape, idxout, center, scale, &preds,
                      this->use_dark);
  result->Reserve(outdata_size);
  result->num_joints = out_data_shape[1];
  result->keypoints.clear();
  for (int i = 0; i < out_data_shape[1]; i++) {
    result->keypoints.push_back({preds[i * 3 + 1], preds[i * 3 + 2]});
    result->scores.push_back(preds[i * 3]);
  }
  return true;
}

bool PPTinyPose::Predict(cv::Mat *im, KeyPointDetectionResult *result) {
  std::vector<float> center = {round(im->cols / 2.0f), round(im->rows / 2.0f)};
  std::vector<float> scale = {static_cast<float>(im->cols),
                              static_cast<float>(im->rows)};
  Mat mat(*im);
  std::vector<FDTensor> processed_data;
  if (!Preprocess(&mat, &processed_data)) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }

  std::vector<FDTensor> infer_result;
  if (!Infer(processed_data, &infer_result)) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }

  if (!Postprocess(infer_result, result, center, scale)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }

  return true;
}

bool PPTinyPose::Predict(cv::Mat *im, KeyPointDetectionResult *result,
                         const DetectionResult &detection_result) {
  std::vector<Mat> crop_imgs;
  std::vector<std::vector<float>> center_bs;
  std::vector<std::vector<float>> scale_bs;
  int crop_imgs_num = 0;
  int box_num = detection_result.boxes.size();
  for (int i = 0; i < box_num; i++) {
    auto box = detection_result.boxes[i];
    auto label_id = detection_result.label_ids[i];
    int channel = im->channels();
    cv::Mat cv_crop_img(0, 0, CV_32SC(channel));
    Mat crop_img(cv_crop_img);
    std::vector<float> rect(box.begin(), box.end());
    std::vector<float> center;
    std::vector<float> scale;
    if (label_id == 0) {
      Mat mat(*im);
      utils::CropImageByBox(mat, &crop_img, rect, &center, &scale);
      center_bs.emplace_back(center);
      scale_bs.emplace_back(scale);
      crop_imgs.emplace_back(crop_img);
      crop_imgs_num += 1;
    }
  }
  for (int i = 0; i < crop_imgs_num; i++) {
    std::vector<FDTensor> processed_data;
    if (!Preprocess(&crop_imgs[i], &processed_data)) {
      FDERROR << "Failed to preprocess input data while using model:"
              << ModelName() << "." << std::endl;
      return false;
    }
    std::vector<FDTensor> infer_result;
    if (!Infer(processed_data, &infer_result)) {
      FDERROR << "Failed to inference while using model:" << ModelName() << "."
              << std::endl;
      return false;
    }
    KeyPointDetectionResult one_cropimg_result;
    if (!Postprocess(infer_result, &one_cropimg_result, center_bs[i],
                     scale_bs[i])) {
      FDERROR << "Failed to postprocess while using model:" << ModelName()
              << "." << std::endl;
      return false;
    }
    if (result->num_joints == -1) {
      result->num_joints = one_cropimg_result.num_joints;
    }
    std::copy(one_cropimg_result.keypoints.begin(),
              one_cropimg_result.keypoints.end(),
              std::back_inserter(result->keypoints));
    std::copy(one_cropimg_result.scores.begin(),
              one_cropimg_result.scores.end(),
              std::back_inserter(result->scores));
  }

  return true;
}

} // namespace keypointdetection
} // namespace vision
} // namespace ultra_infer
