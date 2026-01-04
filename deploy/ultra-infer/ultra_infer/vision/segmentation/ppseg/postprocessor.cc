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
#include "ultra_infer/vision/segmentation/ppseg/postprocessor.h"
#include "ultra_infer/function/cast.h"
#include "yaml-cpp/yaml.h"

namespace ultra_infer {
namespace vision {
namespace segmentation {

PaddleSegPostprocessor::PaddleSegPostprocessor(const std::string &config_file) {
  FDASSERT(ReadFromConfig(config_file),
           "Failed to create PaddleSegPreprocessor.");
  initialized_ = true;
}

bool PaddleSegPostprocessor::ReadFromConfig(const std::string &config_file) {
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file);
  } catch (YAML::BadFile &e) {
    FDERROR << "Failed to load yaml file " << config_file
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  if (cfg["Deploy"]["output_op"]) {
    std::string output_op = cfg["Deploy"]["output_op"].as<std::string>();
    if (output_op == "softmax") {
      is_with_softmax_ = true;
      is_with_argmax_ = false;
    } else if (output_op == "argmax") {
      is_with_softmax_ = false;
      is_with_argmax_ = true;
    } else if (output_op == "none") {
      is_with_softmax_ = false;
      is_with_argmax_ = false;
    } else {
      FDERROR << "Unexcepted output_op operator in deploy.yml: " << output_op
              << "." << std::endl;
      return false;
    }
  }
  return true;
}

bool PaddleSegPostprocessor::SliceOneResultFromBatchInferResults(
    const FDTensor &infer_results, FDTensor *infer_result,
    const std::vector<int64_t> &infer_result_shape, const int64_t &start_idx) {
  int64_t infer_batch = infer_results.shape[0];
  if (infer_batch == 1) {
    *infer_result = infer_results;
    // batch is 1, so ignore
    infer_result->shape = infer_result_shape;
  } else {
    if (infer_results.dtype == FDDataType::FP32) {
      const float_t *infer_results_ptr =
          reinterpret_cast<const float_t *>(infer_results.CpuData()) +
          start_idx;
      infer_result->SetExternalData(
          infer_result_shape, FDDataType::FP32,
          reinterpret_cast<void *>(const_cast<float_t *>(infer_results_ptr)));
    } else if (infer_results.dtype == FDDataType::INT64) {
      const int64_t *infer_results_ptr =
          reinterpret_cast<const int64_t *>(infer_results.CpuData()) +
          start_idx;
      infer_result->SetExternalData(
          infer_result_shape, FDDataType::INT64,
          reinterpret_cast<void *>(const_cast<int64_t *>(infer_results_ptr)));
    } else if (infer_results.dtype == FDDataType::INT32) {
      const int32_t *infer_results_ptr =
          reinterpret_cast<const int32_t *>(infer_results.CpuData()) +
          start_idx;
      infer_result->SetExternalData(
          infer_result_shape, FDDataType::INT32,
          reinterpret_cast<void *>(const_cast<int32_t *>(infer_results_ptr)));
    } else if (infer_results.dtype == FDDataType::UINT8) {
      const uint8_t *infer_results_ptr =
          reinterpret_cast<const uint8_t *>(infer_results.CpuData()) +
          start_idx;
      infer_result->SetExternalData(
          infer_result_shape, FDDataType::UINT8,
          reinterpret_cast<void *>(const_cast<uint8_t *>(infer_results_ptr)));
    } else {
      FDASSERT(
          false,
          "Require the data type for slicing is int64, fp32 or int32, but now "
          "it's %s.",
          Str(infer_results.dtype).c_str())
      return false;
    }
  }
  return true;
}

bool PaddleSegPostprocessor::ProcessWithScoreResult(
    const FDTensor &infer_result, const int64_t &out_num,
    SegmentationResult *result) {
  const uint8_t *argmax_infer_result_buffer = nullptr;
  const float_t *score_infer_result_buffer = nullptr;
  FDTensor argmax_infer_result;
  FDTensor max_score_result;
  std::vector<int64_t> reduce_dim{-1};
  function::ArgMax(infer_result, &argmax_infer_result, -1, FDDataType::UINT8);
  function::Max(infer_result, &max_score_result, reduce_dim);
  score_infer_result_buffer =
      reinterpret_cast<const float_t *>(max_score_result.CpuData());
  std::memcpy(result->score_map.data(), score_infer_result_buffer,
              out_num * sizeof(float_t));

  argmax_infer_result_buffer =
      reinterpret_cast<const uint8_t *>(argmax_infer_result.CpuData());

  std::memcpy(result->label_map.data(), argmax_infer_result_buffer,
              out_num * sizeof(uint8_t));

  return true;
}

bool PaddleSegPostprocessor::ProcessWithLabelResult(
    const FDTensor &infer_result, const int64_t &out_num,
    SegmentationResult *result) {
  if (infer_result.dtype == FDDataType::INT64) {
    const int64_t *infer_result_buffer =
        reinterpret_cast<const int64_t *>(infer_result.CpuData());
    for (int i = 0; i < out_num; i++) {
      result->label_map[i] = static_cast<uint8_t>(*(infer_result_buffer + i));
    }
  } else if (infer_result.dtype == FDDataType::INT32) {
    const int32_t *infer_result_buffer =
        reinterpret_cast<const int32_t *>(infer_result.CpuData());
    for (int i = 0; i < out_num; i++) {
      result->label_map[i] = static_cast<uint8_t>(*(infer_result_buffer + i));
    }
  } else if (infer_result.dtype == FDDataType::UINT8) {
    const uint8_t *infer_result_buffer =
        reinterpret_cast<const uint8_t *>(infer_result.CpuData());
    memcpy(result->label_map.data(), infer_result_buffer,
           out_num * sizeof(uint8_t));
  } else {
    FDASSERT(
        false,
        "Require the data type to process is int64, int32 or uint8, but now "
        "it's %s.",
        Str(infer_result.dtype).c_str());
    return false;
  }
  return true;
}

bool PaddleSegPostprocessor::Run(
    const std::vector<FDTensor> &infer_results,
    std::vector<SegmentationResult> *results,
    const std::map<std::string, std::vector<std::array<int, 2>>> &imgs_info) {
  // PaddleSeg has three types of inference output:
  //     1. output with argmax and without softmax. 3-D matrix N(C)HW, Channel
  //     is batch_size, the element in matrix is classified label_id INT64 type.
  //     2. output without argmax and without softmax. 4-D matrix NCHW, N(batch)
  //     is batch_size, Channel is the num of classes. The element is the logits
  //     of classes FP32 type
  //     3. output without argmax and with softmax. 4-D matrix NCHW, the result
  //     of 2 with softmax layer
  // Xdeploy output:
  //     1. label_map
  //     2. score_map(optional)
  //     3. shape: 2-D HW
  if (!initialized_) {
    FDERROR << "Postprocessor is not initialized." << std::endl;
    return false;
  }

  FDDataType infer_results_dtype = infer_results[0].dtype;
  FDASSERT(infer_results_dtype == FDDataType::INT64 ||
               infer_results_dtype == FDDataType::FP32 ||
               infer_results_dtype == FDDataType::INT32,
           "Require the data type of output is int64, fp32 or int32, but now "
           "it's %s.",
           Str(infer_results_dtype).c_str());

  auto iter_input_imgs_shape_list = imgs_info.find("shape_info");
  FDASSERT(iter_input_imgs_shape_list != imgs_info.end(),
           "Cannot find shape_info from imgs_info.");

  // For Argmax Softmax function to store transformed result below
  FDTensor transform_infer_results;

  int64_t infer_batch = infer_results[0].shape[0];
  int64_t infer_channel = 0;
  int64_t infer_height = 0;
  int64_t infer_width = 0;

  if (is_with_argmax_) {
    // infer_results with argmax
    infer_channel = 1;
    infer_height = infer_results[0].shape[1];
    infer_width = infer_results[0].shape[2];
  } else {
    // infer_results without argmax
    infer_channel = 1;
    infer_height = infer_results[0].shape[2];
    infer_width = infer_results[0].shape[3];
    if (store_score_map_) {
      infer_channel = infer_results[0].shape[1];
      std::vector<int64_t> dim{0, 2, 3, 1};
      function::Transpose(infer_results[0], &transform_infer_results, dim);
      if (!is_with_softmax_ && apply_softmax_) {
        function::Softmax(transform_infer_results, &transform_infer_results, 1);
      }
    } else {
      function::ArgMax(infer_results[0], &transform_infer_results, 1,
                       FDDataType::UINT8);
      infer_results_dtype = transform_infer_results.dtype;
    }
  }

  int64_t infer_chw = infer_channel * infer_height * infer_width;

  results->resize(infer_batch);
  for (int i = 0; i < infer_batch; i++) {
    SegmentationResult *result = &((*results)[i]);
    result->Clear();
    int64_t start_idx = i * infer_chw;

    FDTensor infer_result;
    std::vector<int64_t> infer_result_shape = {infer_height, infer_width,
                                               infer_channel};

    if (is_with_argmax_) {
      SliceOneResultFromBatchInferResults(infer_results[0], &infer_result,
                                          infer_result_shape, start_idx);
    } else {
      SliceOneResultFromBatchInferResults(transform_infer_results,
                                          &infer_result, infer_result_shape,
                                          start_idx);
    }
    bool is_resized = false;
    int input_height = iter_input_imgs_shape_list->second[i][0];
    int input_width = iter_input_imgs_shape_list->second[i][1];
    if (input_height != infer_height || input_width != infer_width) {
      is_resized = true;
    }

    FDMat mat;
    // Resize interpration
    int interpolation = cv::INTER_LINEAR;
    if (is_resized) {
      if (infer_results_dtype == FDDataType::INT64 ||
          infer_results_dtype == FDDataType::INT32) {
        function::Cast(infer_result, &infer_result, FDDataType::UINT8);
        // label map resize with nearest interpolation
        interpolation = cv::INTER_NEAREST;
      }
      mat = std::move(Mat::Create(infer_result, ProcLib::OPENCV));
      Resize::Run(&mat, input_width, input_height, -1.0f, -1.0f, interpolation,
                  false, ProcLib::OPENCV);
      mat.ShareWithTensor(&infer_result);
    }
    result->shape = infer_result.shape;
    // output shape is 2-D HW layout, so out_num = H * W
    int out_num =
        std::accumulate(result->shape.begin(), result->shape.begin() + 2, 1,
                        std::multiplies<int>());

    if (!is_with_argmax_ && store_score_map_) {
      // output with label_map and score_map
      result->contain_score_map = true;
      result->Resize(out_num);
      ProcessWithScoreResult(infer_result, out_num, result);
    } else {
      result->Resize(out_num);
      ProcessWithLabelResult(infer_result, out_num, result);
    }
    // HWC remove C
    result->shape.erase(result->shape.begin() + 2);
  }
  return true;
}
} // namespace segmentation
} // namespace vision
} // namespace ultra_infer
