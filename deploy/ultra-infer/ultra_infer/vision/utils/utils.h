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

#pragma once

#include <opencv2/opencv.hpp>
#include <set>
#include <vector>

#include "ultra_infer/core/fd_tensor.h"
#include "ultra_infer/utils/utils.h"
#include "ultra_infer/vision/common/result.h"

// #include "unsupported/Eigen/CXX11/Tensor"
#include "ultra_infer/function/reduce.h"
#include "ultra_infer/function/softmax.h"
#include "ultra_infer/function/transpose.h"
#include "ultra_infer/vision/common/processors/mat.h"

namespace ultra_infer {
namespace vision {
namespace utils {
// topk sometimes is a very small value
// so this implementation is simple but I don't think it will
// cost too much time
// Also there may be cause problem since we suppose the minimum value is
// -99999999
// Do not use this function on array which topk contains value less than
// -99999999
template <typename T>
std::vector<int32_t> TopKIndices(const T *array, int array_size, int topk) {
  topk = std::min(array_size, topk);
  std::vector<int32_t> res(topk);
  std::set<int32_t> searched;
  for (int32_t i = 0; i < topk; ++i) {
    T min = static_cast<T>(-99999999);
    for (int32_t j = 0; j < array_size; ++j) {
      if (searched.find(j) != searched.end()) {
        continue;
      }
      if (*(array + j) > min) {
        res[i] = j;
        min = *(array + j);
      }
    }
    searched.insert(res[i]);
  }
  return res;
}

void NMS(DetectionResult *output, float iou_threshold = 0.5,
         std::vector<int> *index = nullptr);

void NMS(FaceDetectionResult *result, float iou_threshold = 0.5);

/// Sort DetectionResult/FaceDetectionResult by score
ULTRAINFER_DECL void SortDetectionResult(DetectionResult *result);
ULTRAINFER_DECL void SortDetectionResult(FaceDetectionResult *result);
/// Lex Sort DetectionResult by x(w) & y(h) axis
ULTRAINFER_DECL void LexSortDetectionResultByXY(DetectionResult *result);
/// Lex Sort OCRDet Result by x(w) & y(h) axis
ULTRAINFER_DECL void
LexSortOCRDetResultByXY(std::vector<std::array<int, 8>> *result);

/// L2 Norm / cosine similarity  (for face recognition, ...)
ULTRAINFER_DECL std::vector<float>
L2Normalize(const std::vector<float> &values);

ULTRAINFER_DECL float CosineSimilarity(const std::vector<float> &a,
                                       const std::vector<float> &b,
                                       bool normalized = true);

/** \brief Do face align for model with five points.
 *
 * \param[in] image The original image
 * \param[in] result FaceDetectionResult
 * \param[in] std_landmarks Standard face template
 * \param[in] output_size The size of output mat
 */
ULTRAINFER_DECL std::vector<cv::Mat> AlignFaceWithFivePoints(
    cv::Mat &image, FaceDetectionResult &result,
    std::vector<std::array<float, 2>> std_landmarks = {{38.2946f, 51.6963f},
                                                       {73.5318f, 51.5014f},
                                                       {56.0252f, 71.7366f},
                                                       {41.5493f, 92.3655f},
                                                       {70.7299f, 92.2041f}},
    std::array<int, 2> output_size = {112, 112});

bool CropImageByBox(Mat &src_im, Mat *dst_im, const std::vector<float> &box,
                    std::vector<float> *center, std::vector<float> *scale,
                    const float expandratio = 0.3);

/**
 * Function: for keypoint detection model, fine positioning of keypoints in
 * postprocess
 * Parameters:
 * heatmap: model inference results for keypoint detection models
 * dim: shape information of the inference result
 * coords: coordinates after refined positioning
 * px: px = int(coords[ch * 2] + 0.5) , refer to API
 * detection::GetFinalPredictions py: px = int(coords[ch * 2 + 1] + 0.5), refer
 * to API detection::GetFinalPredictions index: index information of heatmap
 * pixels ch: channel Paper reference: DARK postprocessing, Zhang et al.
 * Distribution-Aware Coordinate Representation for Human Pose Estimation (CVPR
 * 2020).
 */
void DarkParse(const std::vector<float> &heatmap, const std::vector<int> &dim,
               std::vector<float> *coords, const int px, const int py,
               const int index, const int ch);

} // namespace utils
} // namespace vision
} // namespace ultra_infer
