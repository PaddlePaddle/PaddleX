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

#include "ultra_infer/vision/tracking/pptracking/model.h"

#include "ultra_infer/vision/tracking/pptracking/letter_box_resize.h"
#include "yaml-cpp/yaml.h"

namespace ultra_infer {
namespace vision {
namespace tracking {

PPTracking::PPTracking(const std::string &model_file,
                       const std::string &params_file,
                       const std::string &config_file,
                       const RuntimeOption &custom_option,
                       const ModelFormat &model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::PDINFER, Backend::ORT};
  valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;

  initialized = Initialize();
}

bool PPTracking::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile &e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  // Get draw_threshold for visualization
  if (cfg["draw_threshold"].IsDefined()) {
    draw_threshold_ = cfg["draw_threshold"].as<float>();
  } else {
    FDERROR << "Please set draw_threshold." << std::endl;
    return false;
  }
  // Get config for tracker
  if (cfg["tracker"].IsDefined()) {
    if (cfg["tracker"]["conf_thres"].IsDefined()) {
      conf_thresh_ = cfg["tracker"]["conf_thres"].as<float>();
    } else {
      std::cerr << "Please set conf_thres in tracker." << std::endl;
      return false;
    }
    if (cfg["tracker"]["min_box_area"].IsDefined()) {
      min_box_area_ = cfg["tracker"]["min_box_area"].as<float>();
    }
    if (cfg["tracker"]["tracked_thresh"].IsDefined()) {
      tracked_thresh_ = cfg["tracker"]["tracked_thresh"].as<float>();
    }
  }

  processors_.push_back(std::make_shared<BGR2RGB>());
  for (const auto &op : cfg["Preprocess"]) {
    std::string op_name = op["type"].as<std::string>();
    if (op_name == "Resize") {
      bool keep_ratio = op["keep_ratio"].as<bool>();
      auto target_size = op["target_size"].as<std::vector<int>>();
      int interp = op["interp"].as<int>();
      FDASSERT(target_size.size() == 2,
               "Require size of target_size be 2, but now it's %lu.",
               target_size.size());
      if (!keep_ratio) {
        int width = target_size[1];
        int height = target_size[0];
        processors_.push_back(
            std::make_shared<Resize>(width, height, -1.0, -1.0, interp, false));
      } else {
        int min_target_size = std::min(target_size[0], target_size[1]);
        int max_target_size = std::max(target_size[0], target_size[1]);
        std::vector<int> max_size;
        if (max_target_size > 0) {
          max_size.push_back(max_target_size);
          max_size.push_back(max_target_size);
        }
        processors_.push_back(std::make_shared<ResizeByShort>(
            min_target_size, interp, true, max_size));
      }

    } else if (op_name == "LetterBoxResize") {
      auto target_size = op["target_size"].as<std::vector<int>>();
      FDASSERT(target_size.size() == 2,
               "Require size of target_size be 2, but now it's %lu.",
               target_size.size());
      std::vector<float> color{127.0f, 127.0f, 127.0f};
      if (op["fill_value"].IsDefined()) {
        color = op["fill_value"].as<std::vector<float>>();
      }
      processors_.push_back(
          std::make_shared<LetterBoxResize>(target_size, color));
    } else if (op_name == "NormalizeImage") {
      auto mean = op["mean"].as<std::vector<float>>();
      auto std = op["std"].as<std::vector<float>>();
      bool is_scale = true;
      if (op["is_scale"]) {
        is_scale = op["is_scale"].as<bool>();
      }
      std::string norm_type = "mean_std";
      if (op["norm_type"]) {
        norm_type = op["norm_type"].as<std::string>();
      }
      if (norm_type != "mean_std") {
        std::fill(mean.begin(), mean.end(), 0.0);
        std::fill(std.begin(), std.end(), 1.0);
      }
      processors_.push_back(std::make_shared<Normalize>(mean, std, is_scale));
    } else if (op_name == "Permute") {
      // Do nothing, do permute as the last operation
      continue;
      // processors_.push_back(std::make_shared<HWC2CHW>());
    } else if (op_name == "Pad") {
      auto size = op["size"].as<std::vector<int>>();
      auto value = op["fill_value"].as<std::vector<float>>();
      processors_.push_back(std::make_shared<Cast>("float"));
      processors_.push_back(
          std::make_shared<PadToSize>(size[1], size[0], value));
    } else if (op_name == "PadStride") {
      auto stride = op["stride"].as<int>();
      processors_.push_back(
          std::make_shared<StridePad>(stride, std::vector<float>(3, 0)));
    } else {
      FDERROR << "Unexcepted preprocess operator: " << op_name << "."
              << std::endl;
      return false;
    }
  }
  processors_.push_back(std::make_shared<HWC2CHW>());

  FuseTransforms(&processors_);
  return true;
}

bool PPTracking::Initialize() {
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
    return false;
  }
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize ultra_infer backend." << std::endl;
    return false;
  }
  // create JDETracker instance
  jdeTracker_ = std::unique_ptr<JDETracker>(new JDETracker);
  return true;
}

bool PPTracking::Predict(cv::Mat *img, MOTResult *result) {
  Mat mat(*img);
  std::vector<FDTensor> input_tensors;

  if (!Preprocess(&mat, &input_tensors)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }
  std::vector<FDTensor> output_tensors;
  if (!Infer(input_tensors, &output_tensors)) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }

  if (!Postprocess(output_tensors, result)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }
  return true;
}

bool PPTracking::Preprocess(Mat *mat, std::vector<FDTensor> *outputs) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();

  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }

  //  LetterBoxResize(mat);
  //  Normalize::Run(mat,mean_,scale_,is_scale_);
  //  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");

  outputs->resize(3);
  // image_shape
  (*outputs)[0].Allocate({1, 2}, FDDataType::FP32, InputInfoOfRuntime(0).name);
  float *shape = static_cast<float *>((*outputs)[0].MutableData());
  shape[0] = mat->Height();
  shape[1] = mat->Width();
  // image
  (*outputs)[1].name = InputInfoOfRuntime(1).name;
  mat->ShareWithTensor(&((*outputs)[1]));
  (*outputs)[1].ExpandDim(0);
  // scale
  (*outputs)[2].Allocate({1, 2}, FDDataType::FP32, InputInfoOfRuntime(2).name);
  float *scale = static_cast<float *>((*outputs)[2].MutableData());
  scale[0] = mat->Height() * 1.0 / origin_h;
  scale[1] = mat->Width() * 1.0 / origin_w;
  return true;
}

void FilterDets(const float conf_thresh, const cv::Mat &dets,
                std::vector<int> *index) {
  for (int i = 0; i < dets.rows; ++i) {
    float score = *dets.ptr<float>(i, 4);
    if (score > conf_thresh) {
      index->push_back(i);
    }
  }
}

bool PPTracking::Postprocess(std::vector<FDTensor> &infer_result,
                             MOTResult *result) {
  auto bbox_shape = infer_result[0].shape;
  auto bbox_data = static_cast<float *>(infer_result[0].Data());

  auto emb_shape = infer_result[1].shape;
  auto emb_data = static_cast<float *>(infer_result[1].Data());

  cv::Mat dets(bbox_shape[0], 6, CV_32FC1, bbox_data);
  cv::Mat emb(bbox_shape[0], emb_shape[1], CV_32FC1, emb_data);

  result->Clear();
  std::vector<Track> tracks;
  std::vector<int> valid;
  FilterDets(conf_thresh_, dets, &valid);
  cv::Mat new_dets, new_emb;
  for (int i = 0; i < valid.size(); ++i) {
    new_dets.push_back(dets.row(valid[i]));
    new_emb.push_back(emb.row(valid[i]));
  }
  jdeTracker_->update(new_dets, new_emb, &tracks);
  if (tracks.size() == 0) {
    std::array<int, 4> box = {
        int(*dets.ptr<float>(0, 0)), int(*dets.ptr<float>(0, 1)),
        int(*dets.ptr<float>(0, 2)), int(*dets.ptr<float>(0, 3))};
    result->boxes.push_back(box);
    result->ids.push_back(1);
    result->scores.push_back(*dets.ptr<float>(0, 4));
  } else {
    std::vector<Track>::iterator titer;
    for (titer = tracks.begin(); titer != tracks.end(); ++titer) {
      if (titer->score < tracked_thresh_) {
        continue;
      } else {
        float w = titer->ltrb[2] - titer->ltrb[0];
        float h = titer->ltrb[3] - titer->ltrb[1];
        bool vertical = w / h > 1.6;
        float area = w * h;
        if (area > min_box_area_ && !vertical) {
          std::array<int, 4> box = {int(titer->ltrb[0]), int(titer->ltrb[1]),
                                    int(titer->ltrb[2]), int(titer->ltrb[3])};
          result->boxes.push_back(box);
          result->ids.push_back(titer->id);
          result->scores.push_back(titer->score);
        }
      }
    }
  }
  if (!is_record_trail_)
    return true;
  int nums = result->boxes.size();
  for (int i = 0; i < nums; i++) {
    float center_x = (result->boxes[i][0] + result->boxes[i][2]) / 2;
    float center_y = (result->boxes[i][1] + result->boxes[i][3]) / 2;
    int id = result->ids[i];
    recorder_->Add(id, {int(center_x), int(center_y)});
  }
  return true;
}

void PPTracking::BindRecorder(TrailRecorder *recorder) {
  recorder_ = recorder;
  is_record_trail_ = true;
}

void PPTracking::UnbindRecorder() {
  is_record_trail_ = false;
  std::map<int, std::vector<std::array<int, 2>>>::iterator iter;
  for (iter = recorder_->records.begin(); iter != recorder_->records.end();
       iter++) {
    iter->second.clear();
    iter->second.shrink_to_fit();
  }
  recorder_->records.clear();
  std::map<int, std::vector<std::array<int, 2>>>().swap(recorder_->records);
  recorder_ = nullptr;
}

} // namespace tracking
} // namespace vision
} // namespace ultra_infer
