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
#pragma once
#include <memory>
#include <string>
#include <thread> // NOLINT
#include <vector>

#include "model_deploy/common/include/deploy_delacre.h"
#include "model_deploy/common/include/model_factory.h"
#include "model_deploy/engine/include/engine.h"

namespace PaddleDeploy {
class PD_INFER_DECL MultiGPUModel {
 private:
  std::vector<std::shared_ptr<Model>> models_;

 public:
  bool Init(const std::string& model_type,
            const std::string& cfg_file, size_t gpu_num = 1) {
    models_.clear();
    for (auto i = 0; i < gpu_num; ++i) {
      Model* model = PaddleDeploy::ModelFactory::CreateObject(model_type);

      if (!model) {
        std::cerr << "no model_type: " << model_type << std::endl;
        return false;
      }

      std::cerr << i + 1 << " model start init" << std::endl;

      if (!model->Init(cfg_file)) {
        std::cerr << "model Init error" << std::endl;
        return false;
      }

      models_.push_back(std::shared_ptr<Model>(model));
    }
    return true;
  }

  bool PaddleEngineInit(PaddleEngineConfig engine_config,
                        const std::vector<int> gpu_ids) {
    if (gpu_ids.size() != models_.size()) {
      std::cerr << "Paddle Engine Init gpu_ids != MultiGPUModel Init gpu_num"
                << gpu_ids.size() << " != " << models_.size()
                << std::endl;
      return false;
    }
    engine_config.use_gpu = true;
    for (auto i = 0; i < gpu_ids.size(); ++i) {
      engine_config.gpu_id = gpu_ids[i];
      if (!models_[i]->PaddleEngineInit(engine_config)) {
        std::cerr << "Paddle Engine Init error:" << gpu_ids[i] << std::endl;
        return false;
      }
    }
    return true;
  }

  bool Predict(const std::vector<cv::Mat>& imgs,
               std::vector<Result>* results,
               int thread_num = 1) {
    results->clear();
    int model_num = models_.size();
    if (model_num <= 0) {
      std::cerr << "Please Init before Predict!" << std::endl;
      return false;
    }

    int imgs_size = imgs.size();
    int start = 0;
    std::vector<std::thread> threads;
    std::vector<std::vector<cv::Mat>> split_imgs;
    std::vector<std::vector<Result>> model_results;
    for (int i = 0; i < model_num; ++i) {
      int img_num = static_cast<int>(imgs_size / model_num);
      if (i < imgs_size % model_num) {
        img_num += 1;
      } else if (img_num <= 0) {
        // imgs.size < model_.size
        break;
      }
      std::vector<cv::Mat> new_imgs(imgs.begin() + start,
                                    imgs.begin() + start + img_num);
      split_imgs.push_back(new_imgs);
      start += img_num;
    }

    model_results.resize(split_imgs.size());
    for (int i = 0; i < split_imgs.size(); ++i) {
      threads.push_back(std::thread(&PaddleDeploy::Model::Predict, models_[i],
                                    std::ref(split_imgs[i]),
                                    &model_results[i], thread_num));
    }

    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    // merge result
    for (auto model_result : model_results) {
      results->insert(results->end(),
                      model_result.begin(), model_result.end());
    }

    return true;
  }
};
}  // namespace PaddleDeploy
