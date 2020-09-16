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

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "yaml-cpp/yaml.h"

#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

#include "paddle_inference_api.h"  // NOLINT

#include "config_parser.h"  // NOLINT
#include "results.h"  // NOLINT
#include "transforms.h"  // NOLINT

#ifdef WITH_ENCRYPTION
#include "paddle_model_decrypt.h"  // NOLINT
#include "model_code.h"  // NOLINT
#endif

namespace PaddleX {

/*
 * @brief
 * This class encapsulates all necessary proccess steps of model infering, which
 * include image matrix preprocessing, model predicting and results postprocessing.
 * The entire process of model infering can be simplified as below:
 * 1. preprocess image matrix (resize, padding, ......)
 * 2. model infer
 * 3. postprocess the results which generated from model infering
 *
 * @example
 *  PaddleX::Model cls_model;
 *  // initialize model configuration
 *  cls_model.Init(cls_model_dir, use_gpu, use_trt, gpu_id, encryption_key);
 *  // define a Classification result object
 *  PaddleX::ClsResult cls_result;
 *  // get image matrix from image file
 *  cv::Mat im = cv::imread(image_file_path, 1);
 *  cls_model.predict(im, &cls_result);
 * */
class Model {
 public:
  /*
   * @brief
   * This method aims to initialize the model configuration
   *
   * @param model_dir: the directory which contains model.yml
   * @param use_gpu: use gpu or not when infering
   * @param use_trt: use Tensor RT or not when infering
   * @param use_mkl: use mkl or not when infering
   * @param mkl_thread_num: number of threads for mkldnn when infering
   * @param gpu_id: the id of gpu when infering with using gpu
   * @param key: the key of encryption when using encrypted model
   * @param use_ir_optim: use ir optimization when infering
   * */
  void Init(const std::string& model_dir,
            bool use_gpu = false,
            bool use_trt = false,
            bool use_mkl = true,
            int mkl_thread_num = 4,
            int gpu_id = 0,
            std::string key = "",
            bool use_ir_optim = true) {
    create_predictor(
                     model_dir,
                     use_gpu,
                     use_trt,
                     use_mkl,
                     mkl_thread_num,
                     gpu_id,
                     key,
                     use_ir_optim);
  }
  void create_predictor(const std::string& model_dir,
                        bool use_gpu = false,
                        bool use_trt = false,
                        bool use_mkl = true,
                        int mkl_thread_num = 4,
                        int gpu_id = 0,
                        std::string key = "",
                        bool use_ir_optim = true);

  /*
   * @brief
   * This method aims to load model configurations which include
   * transform steps and label list
   *
   * @param yaml_input:  model configuration string
   * @return true if load configuration successfully
   * */
  bool load_config(const std::string& yaml_input);

  /*
   * @brief
   * This method aims to transform single image matrix, the result will be
   * returned at second parameter.
   *
   * @param input_im: single image matrix to be transformed
   * @param blob: the raw data of single image matrix after transformed
   * @return true if preprocess image matrix successfully
   * */
  bool preprocess(const cv::Mat& input_im, ImageBlob* blob);

  /*
   * @brief
   * This method aims to transform mutiple image matrixs, the result will be
   * returned at second parameter.
   *
   * @param input_im_batch: a batch of image matrixs to be transformed
   * @param blob_blob: raw data of a batch of image matrixs after transformed
   * @param thread_num: the number of preprocessing threads,
   *                    each thread run preprocess on single image matrix
   * @return true if preprocess a batch of image matrixs successfully
   * */
  bool preprocess(const std::vector<cv::Mat> &input_im_batch,
                  std::vector<ImageBlob> *blob_batch,
                  int thread_num = 1);

  /*
   * @brief
   * This method aims to execute classification model prediction on single image matrix,
   * the result will be returned at second parameter.
   *
   * @param im: single image matrix to be predicted
   * @param result: classification prediction result data after postprocessed
   * @return true if predict successfully
   * */
  bool predict(const cv::Mat& im, ClsResult* result);

  /*
   * @brief
   * This method aims to execute classification model prediction on a batch of image matrixs,
   * the result will be returned at second parameter.
   *
   * @param im: a batch of image matrixs to be predicted
   * @param results: a batch of classification prediction result data after postprocessed
   * @param thread_num: the number of predicting threads, each thread run prediction
   *                    on single image matrix
   * @return true if predict successfully
   * */
  bool predict(const std::vector<cv::Mat> &im_batch,
               std::vector<ClsResult> *results,
               int thread_num = 1);

  /*
   * @brief
   * This method aims to execute detection or instance segmentation model prediction
   * on single image matrix, the result will be returned at second parameter.
   *
   * @param im: single image matrix to be predicted
   * @param result: detection or instance segmentation prediction result data after postprocessed
   * @return true if predict successfully
   * */
  bool predict(const cv::Mat& im, DetResult* result);

  /*
   * @brief
   * This method aims to execute detection or instance segmentation model prediction
   * on a batch of image matrixs, the result will be returned at second parameter.
   *
   * @param im: a batch of image matrix to be predicted
   * @param result: detection or instance segmentation prediction result data after postprocessed
   * @param thread_num: the number of predicting threads, each thread run prediction
   *                    on single image matrix
   * @return true if predict successfully
   * */
  bool predict(const std::vector<cv::Mat> &im_batch,
               std::vector<DetResult> *results,
               int thread_num = 1);

  /*
   * @brief
   * This method aims to execute segmentation model prediction on single image matrix,
   * the result will be returned at second parameter.
   *
   * @param im: single image matrix to be predicted
   * @param result: segmentation prediction result data after postprocessed
   * @return true if predict successfully
   * */
  bool predict(const cv::Mat& im, SegResult* result);

  /*
   * @brief
   * This method aims to execute segmentation model prediction on a batch of image matrix,
   * the result will be returned at second parameter.
   *
   * @param im: a batch of image matrix to be predicted
   * @param result: segmentation prediction result data after postprocessed
   * @param thread_num: the number of predicting threads, each thread run prediction
   *                    on single image matrix
   * @return true if predict successfully
   * */
  bool predict(const std::vector<cv::Mat> &im_batch,
               std::vector<SegResult> *results,
               int thread_num = 1);

  // model type, include 3 type: classifier, detector, segmenter
  std::string type;
  // model name, such as FasterRCNN, YOLOV3 and so on.
  std::string name;
  std::map<int, std::string> labels;
  // transform(preprocessing) pipeline manager
  Transforms transforms_;
  // single input preprocessed data
  ImageBlob inputs_;
  // batch input preprocessed data
  std::vector<ImageBlob> inputs_batch_;
  // raw data of predicting results
  std::vector<float> outputs_;
  // a predictor which run the model predicting
  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  // input channel
  int input_channel_;
};
}  // namespace PaddleX
