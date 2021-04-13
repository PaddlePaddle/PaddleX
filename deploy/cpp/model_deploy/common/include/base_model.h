//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "model_deploy/common/include/base_postprocess.h"
#include "model_deploy/common/include/base_preprocess.h"
#include "model_deploy/engine/include/engine.h"
#include "model_deploy/common/include/output_struct.h"
#include "paddle_inference_api.h"
#include "yaml-cpp/yaml.h"

namespace PaddleDeploy {

class Model {
 private:
  const std::string model_type_;

 public:
  /*store the data after the YAML file has been parsed */
  YAML::Node yaml_config_;
  /* preprocess */
  std::shared_ptr<BasePreProcess> preprocess_;
  /* inference */
  std::shared_ptr<InferEngine> infer_engine_;
  /* postprocess */
  std::shared_ptr<BasePostProcess> postprocess_;
  /*postprocess result*/
  std::vector<Result> results_;

  Model() {}

  // Init model_type.
  explicit Model(const std::string model_type) : model_type_(model_type) {}

  virtual void Init(const std::string& cfg_file, bool use_cpu_nms) {
    YamlConfigInit(cfg_file);
    PreProcessInit();
    PostProcessInit(use_cpu_nms);
  }

  virtual void YamlConfigInit(const std::string& cfg_file) {
    YAML::Node yaml_config_ = YAML::LoadFile(cfg_file);
  }

  virtual void PreProcessInit() {
    preprocess_ = nullptr;
  }

  void PaddleEngineInit(const std::string& model_filename,
                        const std::string& params_filename,
                        bool use_gpu = false,
                        int gpu_id = 0,
                        bool use_mkl = true);

  virtual void PostProcessInit(bool use_cpu_nms) {
    postprocess_ = nullptr;
  }

  virtual bool Predict(const std::vector<cv::Mat>& imgs, int thread_num = 1) {
    if (!preprocess_ || !postprocess_ || !infer_engine_) {
      std::cerr << "No init,cann't predict" << std::endl;
      return false;
    }

    std::vector<cv::Mat> imgs_clone(imgs.size());
    for (auto i = 0; i < imgs.size(); ++i) {
      imgs[i].copyTo(imgs_clone[i]);
    }

    std::vector<ShapeInfo> shape_infos;
    std::vector<DataBlob> inputs;
    std::vector<DataBlob> outputs;

    preprocess_->Run(&imgs_clone, &inputs, &shape_infos, thread_num);
    infer_engine_->Infer(inputs, &outputs);
    postprocess_->Run(outputs, shape_infos, &results_, thread_num);
    return true;
  }

  virtual void PrintResult() {
    for (auto i = 0; i < results_.size(); ++i) {
      std::cout << "result for sample " << i << std::endl;
      std::cout << results_[i] << std::endl;
    }
  }

  virtual void PrePrecess(const std::vector<cv::Mat>& imgs,
                          std::vector<DataBlob>* inputs,
                          std::vector<ShapeInfo>* shape_infos,
                          int thread_num = 1) {
    if (!preprocess_) {
      std::cerr << "No PrePrecess, No pre Init. model_type="
                << model_type_ << std::endl;
      return;
    }

    std::vector<cv::Mat> imgs_clone(imgs.size());
    for (auto i = 0; i < imgs.size(); ++i) {
      imgs[i].copyTo(imgs_clone[i]);
    }


    preprocess_->Run(&imgs_clone, inputs, shape_infos, thread_num);
  }

  virtual void Infer(const std::vector<DataBlob>& inputs,
                     std::vector<DataBlob>* outputs) {
    infer_engine_->Infer(inputs, outputs);
  }

  virtual void PostPrecess(const std::vector<DataBlob>& outputs,
                           const std::vector<ShapeInfo>& shape_infos,
                           int thread_num = 1) {
    if (!postprocess_) {
      std::cerr << "No PostPrecess, No post Init. model_type="
                << model_type_ << std::endl;
      return;
    }
    postprocess_->Run(outputs, shape_infos, &results_, thread_num);
  }
};

}  // namespace PaddleDeploy
