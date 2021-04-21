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
  bool Init(const std::string& model_type, const std::string& cfg_file,
            const std::string& model_filename,
            const std::string& params_filename, const std::vector<int> gpu_ids,
            bool use_gpu, bool use_mkl) {
    int model_num = static_cast<int>(gpu_ids.size());
    for (auto i = 0; i < model_num; ++i) {
      std::shared_ptr<Model> model =
          PaddleDeploy::ModelFactory::CreateObject(model_type);

      if (!model) {
        std::cout << "no model_type: " << model_type << std::endl;
        return false;
      }

      if (!model->Init(cfg_file)) {
        std::cerr << "model Init error" << std::endl;
        return false;
      }

      if (!model->PaddleEngineInit(model_filename, params_filename, use_gpu,
                                   gpu_ids[i], use_mkl)) {
        std::cerr << "Paddle Engine Init error" << std::endl;
        return false;
      }
      models_.push_back(model);
    }
  }

  bool Predict(const std::vector<cv::Mat>& imgs, int thread_num = 1,
               int imgs_start = 0, int imgs_end = -1) {
    run_id_.clear();

    if (imgs_end < 0) {
      imgs_end = imgs.size();
    }
    int imgs_size = imgs_end - imgs_start;
    if (imgs_size == 0) {
      std::cerr << "predict no image !" << std::endl;
      return true;
    }
    int model_num = models_.size();
    int remainder = imgs_size % model_num;
    int thread_imgs_size = static_cast<int>(imgs_size / model_num);

    int start = 0;
    int img_num;
    std::vector<std::thread> threads;
    for (int i = 0; i < model_num; ++i) {
      img_num = (i < remainder) ? thread_imgs_size + 1 : thread_imgs_size;
      run_id_.push_back(i);
      threads.push_back(std::thread(&PaddleDeploy::Model::Predict, models_[i],
                                    std::ref(imgs), thread_num, start,
                                    start + img_num));
      start += img_num;
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
      std::cout << "result for sample " << i << std::endl;
      for (auto result : models_[id]->results_) {
        std::cout << result << std::endl;
        i += 1;
      }
    }
  }
};
}  // namespace PaddleDeploy