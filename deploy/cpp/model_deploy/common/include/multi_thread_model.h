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
#include <future>
#include <queue>

#include "model_deploy/common/include/deploy_declare.h"
#include "model_deploy/common/include/model_factory.h"
#include "model_deploy/engine/include/engine.h"

namespace PaddleDeploy {
class PD_INFER_DECL MultiThreadModel {
  class ThreadWorker {
    private:
    int m_id;
    MultiThreadModel* m_model;

  public:
    ThreadWorker(MultiThreadModel* model, const int id) 
      : m_model(model), m_id(id) {
    }

    void operator()() {
      while (!m_model->m_shutdown) {
        std::vector<cv::Mat> imgs;
        std::vector<Result>* result;
        std::shared_ptr<std::promise<bool>> notify;
        {
          std::unique_lock<std::mutex> lock(m_model->m_conditional_mutex);
          if (m_model->m_input.empty()) {
            m_model->m_conditional_lock.wait(lock);
          }
          // std::unique_lock<std::mutex> lock(m_model->queue_mutex);
          imgs = std::move(m_model->m_input.front());
          m_model->m_input.pop();
          result = std::move(m_model->m_result.front());
          m_model->m_result.pop();
          notify = std::move(m_model->m_notify.front());
          m_model->m_notify.pop();
        }
        m_model->models_[m_id]->Predict(imgs, result, 1);
        notify->set_value(true);
        notify.reset();
      }
    }
  };
 private:
  bool m_shutdown;
  std::vector<std::thread> m_threads;
  std::queue<std::vector<cv::Mat>> m_input;
  std::queue<std::vector<Result>*> m_result;
  std::queue<std::shared_ptr<std::promise<bool>>> m_notify;
  std::mutex m_conditional_mutex;
  //std::mutex queue_mutex;
  std::condition_variable m_conditional_lock;
  std::vector<std::shared_ptr<Model>> models_;

 public:
  bool Init(const std::string& model_type,
            const std::string& cfg_file, size_t thread_num = 1) {
    models_.clear();
    m_threads.clear();
    for (auto i = 0; i < thread_num; ++i) {
      Model* model = PaddleDeploy::ModelFactory::CreateObject(model_type);

      if (!model) {
        std::cerr << "no model_type: " << model_type << std::endl;
        return false;
      }

      std::cout << i + 1 << " model start init" << std::endl;

      if (!model->Init(cfg_file)) {
        std::cerr << "model Init error" << std::endl;
        return false;
      }
      models_.push_back(std::shared_ptr<Model>(model));
      m_threads.push_back(std::thread(ThreadWorker(this, i)));
    }
    return true;
  }

  bool PaddleEngineInit(PaddleEngineConfig engine_config,
                        const std::vector<int> gpu_ids) {
    if (engine_config.use_gpu) {
      if (gpu_ids.size() != models_.size()) {
      std::cerr << "Paddle Engine Init gpu_ids != MultiThreadModel Init thread_num"
                << gpu_ids.size() << " != " << models_.size()
                << std::endl;
      return false;
      }
      for (auto i = 0; i < gpu_ids.size(); ++i) {
        engine_config.gpu_id = gpu_ids[i];
        if (!models_[i]->PaddleEngineInit(engine_config)) {
          std::cerr << "Paddle Engine Init error:" << gpu_ids[i] << std::endl;
          return false;
        }
      }
    } else {
      for (auto i = 0; i < models_.size(); ++i) {
        if (!models_[i]->PaddleEngineInit(engine_config)) {
          std::cerr << "Paddle Engine cpu Init error:" << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  std::future<bool> add_predict_task(const std::vector<cv::Mat>& imgs,
               std::vector<Result>* results){
    std::unique_lock<std::mutex> lock(this->m_conditional_mutex);
    std::promise<bool> notify1;
    std::future<bool> future = notify1.get_future();
    auto notify = std::make_shared<std::promise<bool>>(std::move(notify1));
    m_input.push(imgs);
    m_result.push(results);
    m_notify.push(notify);
    m_conditional_lock.notify_one();
    return future;
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
    std::vector<std::vector<cv::Mat>> split_imgs;
    std::vector<std::vector<Result>> model_results;
    std::vector<std::future<bool>> futures;
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
    futures.resize(split_imgs.size());
    for (int i = 0; i < split_imgs.size(); ++i) {
      futures[i] = add_predict_task(split_imgs[i], &model_results[i]);
    }

    for (int i = 0; i < split_imgs.size(); ++i) {
      futures[i].get();
      results->insert(results->end(),
                      model_results[i].begin(), model_results[i].end());
    }
    return true;
  }
};
}  // namespace PaddleDeploy
