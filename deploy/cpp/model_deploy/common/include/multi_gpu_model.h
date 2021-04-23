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
#include <memory>
#include <string>
#include <thread>

#include "common/include/model_factory.h"

namespace PaddleDeploy {
class MultiGPUModel {
 private:
  std::vector<std::shared_ptr<Model>> models_;
  //record predict id
  std::vector<u_int> run_id_;

 public:
  bool Init(const std::string& model_type,
            const std::string& cfg_file, size_t gpu_num = 1) {
    models_.clear();
    for (auto i = 0; i < gpu_num; ++i) {
      std::shared_ptr<Model> model =
          PaddleDeploy::ModelFactory::CreateObject(model_type);

      if (!model) {
        std::cout << "no model_type: " << model_type << std::endl;
        return false;
      }

      std::cout << i + 1 << " model start init" << std::endl;

      if (!model->Init(cfg_file)) {
        std::cerr << "model Init error" << std::endl;
        return false;
      }

      models_.push_back(model);
    }
    return true;
  }

  bool PaddleEngineInit(const std::string& model_filename,
                        const std::string& params_filename,
                        const std::vector<int> gpu_ids,
                        bool use_gpu = false, bool use_mkl = true) {
    for (auto i = 0; i < gpu_ids.size(); ++i){
      if (!models_[i]->PaddleEngineInit(model_filename,
                                        params_filename,
                                        use_gpu, gpu_ids[i],
                                        use_mkl)) {
        std::cerr << "Paddle Engine Init error:" << gpu_ids[i] << std::endl;
        return false;
      }
    }
    return true;
  }

  bool Predict(const std::vector<cv::Mat>& imgs, int thread_num = 1) {
    run_id_.clear();
    int model_num = models_.size();
    if (model_num <= 0) {
      std::cerr << "Please Init before Predict!" << std::endl;
      return false;
    }

    int imgs_size = imgs.size();
    if (imgs_size == 1) {
      models_[0]->Predict(imgs);
      run_id_.push_back(0);
      return true;
    }

    int start = 0;
    std::vector<std::thread> threads;
    std::vector<std::vector<cv::Mat>> split_imgs;
    for (int i = 0; i < model_num; ++i) {
      int img_num = static_cast<int>(imgs_size / model_num);
      if (i < imgs_size % model_num) {
        img_num += 1;
      } else if (img_num <= 0) {
        //imgs.size < model_.size
        break;
      }
      run_id_.push_back(i);
      std::vector<cv::Mat> new_imgs(imgs.begin() + start,
                                    imgs.begin() + start + img_num);
      split_imgs.push_back(new_imgs);
      start += img_num;
    }

    for (int i = 0; i < model_num; ++i) {
      threads.push_back(std::thread(&PaddleDeploy::Model::Predict, models_[i],
                                    std::ref(split_imgs[i]), thread_num));
    }

    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    return true;
  }

  void PrintResult() {
    int i = 0;
    for (u_int id : run_id_) {
      std::cout << "image " << i << std::endl;
      for (auto result : models_[id]->results_) {
        std::cout << "boxes num:"
                  << result.det_result->boxes.size()
                  << std::endl;
        std::cout << result << std::endl;
        i += 1;
      }
    }
  }
};
}  // namespace PaddleDeploy