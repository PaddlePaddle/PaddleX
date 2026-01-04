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

#include "opencv2/imgproc/imgproc.hpp"
#include "ultra_infer/vision/common/result.h"
#include "ultra_infer/vision/tracking/pptracking/model.h"

namespace ultra_infer {
/** \brief All C++ UltraInfer Vision Models APIs are defined inside this
 * namespace
 *
 */
namespace vision {

class ULTRAINFER_DECL Visualize {
public:
  static int num_classes_;
  static std::vector<int> color_map_;
  static const std::vector<int> &GetColorMap(int num_classes = 1000);
  static cv::Mat VisDetection(const cv::Mat &im, const DetectionResult &result,
                              float score_threshold = 0.0, int line_size = 1,
                              float font_size = 0.5f);
  static cv::Mat VisPerception(const cv::Mat &im,
                               const PerceptionResult &result,
                               const std::string &config_file,
                               float score_threshold = 0.0, int line_size = 1,
                               float font_size = 0.5f);
  static cv::Mat VisFaceDetection(const cv::Mat &im,
                                  const FaceDetectionResult &result,
                                  int line_size = 1, float font_size = 0.5f);
  static cv::Mat VisSegmentation(const cv::Mat &im,
                                 const SegmentationResult &result);
  static cv::Mat VisMattingAlpha(const cv::Mat &im, const MattingResult &result,
                                 bool remove_small_connected_area = false);
  static cv::Mat RemoveSmallConnectedArea(const cv::Mat &alpha_pred,
                                          float threshold);
  static cv::Mat
  SwapBackgroundMatting(const cv::Mat &im, const cv::Mat &background,
                        const MattingResult &result,
                        bool remove_small_connected_area = false);
  static cv::Mat SwapBackgroundSegmentation(const cv::Mat &im,
                                            const cv::Mat &background,
                                            int background_label,
                                            const SegmentationResult &result);
  static cv::Mat VisOcr(const cv::Mat &srcimg, const OCRResult &ocr_result);
  static cv::Mat VisCURVEOcr(const cv::Mat &srcimg,
                             const OCRCURVEResult &ocr_result);
};

std::vector<int> GenerateColorMap(int num_classes = 1000);
cv::Mat RemoveSmallConnectedArea(const cv::Mat &alpha_pred, float threshold);
/** \brief Show the visualized results for detection models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] result the result produced by model
 * \param[in] score_threshold threshold for result scores, the bounding box will
 * not be shown if the score is less than score_threshold \param[in] line_size
 * line size for bounding boxes \param[in] font_size font size for text \return
 * cv::Mat type stores the visualized results
 */
ULTRAINFER_DECL cv::Mat VisDetection(const cv::Mat &im,
                                     const DetectionResult &result,
                                     float score_threshold = 0.0,
                                     int line_size = 1, float font_size = 0.5f);
/** \brief Show the visualized results with custom labels for detection models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] result the result produced by model
 * \param[in] labels the visualized result will show the bounding box contain
 * class label \param[in] score_threshold threshold for result scores, the
 * bounding box will not be shown if the score is less than score_threshold
 * \param[in] line_size line size for bounding boxes
 * \param[in] font_size font size for text
 * \param[in] font_color font color for bounding text
 * \param[in] font_thickness font thickness for text
 * \return cv::Mat type stores the visualized results
 */
ULTRAINFER_DECL cv::Mat VisDetection(
    const cv::Mat &im, const DetectionResult &result,
    const std::vector<std::string> &labels, float score_threshold = 0.0,
    int line_size = 1, float font_size = 0.5f,
    std::vector<int> font_color = {255, 255, 255}, int font_thickness = 1);

/** \brief Show the visualized results with custom labels for detection models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] result the result produced by model
 * \param[in] labels the visualized result will show the bounding box contain
 * class label \param[in] score_threshold threshold for result scores, the
 * bounding box will not be shown if the score is less than score_threshold
 * \param[in] line_size line size for bounding boxes
 * \param[in] font_size font size for text
 * \return cv::Mat type stores the visualized results
 */
ULTRAINFER_DECL cv::Mat
VisPerception(const cv::Mat &im, const PerceptionResult &result,
              const std::string &config_file, float score_threshold = 0.0,
              int line_size = 1, float font_size = 0.5f);
/** \brief Show the visualized results for classification models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] result the result produced by model
 * \param[in] top_k the length of return values, e.g., if topk==2, the result
 * will include the 2 most possible class label for input image. \param[in]
 * score_threshold threshold for top_k scores, the class will not be shown if
 * the score is less than score_threshold \param[in] font_size font size \return
 * cv::Mat type stores the visualized results
 */
ULTRAINFER_DECL cv::Mat VisClassification(const cv::Mat &im,
                                          const ClassifyResult &result,
                                          int top_k = 5,
                                          float score_threshold = 0.0f,
                                          float font_size = 0.5f);
/** \brief Show the visualized results with custom labels for classification
 * models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] result the result produced by model
 * \param[in] labels custom labels for user, the visualized result will show the
 * corresponding custom labels \param[in] top_k the length of return values,
 * e.g., if topk==2, the result will include the 2 most possible class label for
 * input image. \param[in] score_threshold threshold for top_k scores, the class
 * will not be shown if the score is less than score_threshold \param[in]
 * font_size font size \return cv::Mat type stores the visualized results
 */
ULTRAINFER_DECL cv::Mat
VisClassification(const cv::Mat &im, const ClassifyResult &result,
                  const std::vector<std::string> &labels, int top_k = 5,
                  float score_threshold = 0.0f, float font_size = 0.5f);
/** \brief Show the visualized results for face detection models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] result the result produced by model
 * \param[in] line_size line size for bounding boxes
 * \param[in] font_size font size for text
 * \return cv::Mat type stores the visualized results
 */
ULTRAINFER_DECL cv::Mat VisFaceDetection(const cv::Mat &im,
                                         const FaceDetectionResult &result,
                                         int line_size = 1,
                                         float font_size = 0.5f);
/** \brief Show the visualized results for face alignment models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] result the result produced by model
 * \param[in] line_size line size for circle point
 * \return cv::Mat type stores the visualized results
 */
ULTRAINFER_DECL cv::Mat VisFaceAlignment(const cv::Mat &im,
                                         const FaceAlignmentResult &result,
                                         int line_size = 1);
/** \brief Show the visualized results for segmentation models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] result the result produced by model
 * \param[in] weight transparent weight of visualized result image
 * \return cv::Mat type stores the visualized results
 */
ULTRAINFER_DECL cv::Mat VisSegmentation(const cv::Mat &im,
                                        const SegmentationResult &result,
                                        float weight = 0.5);
/** \brief Show the visualized results for matting models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] result the result produced by model
 * \param[in] transparent_background if transparent_background==true, the
 * background will with transparent color \param[in] transparent_threshold since
 * the alpha value in MattringResult is a float between [0, 1],
 * transparent_threshold is used to filter background pixel \param[in]
 * remove_small_connected_area if remove_small_connected_area==true, the
 * visualized result will not include the small connected areas \return cv::Mat
 * type stores the visualized results
 */
ULTRAINFER_DECL cv::Mat VisMatting(const cv::Mat &im,
                                   const MattingResult &result,
                                   bool transparent_background = false,
                                   float transparent_threshold = 0.999,
                                   bool remove_small_connected_area = false);
/** \brief Show the visualized results for Ocr models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] result the result produced by model
 * \return cv::Mat type stores the visualized results
 */
ULTRAINFER_DECL cv::Mat VisOcr(const cv::Mat &im, const OCRResult &ocr_result,
                               const float score_threshold = 0);
ULTRAINFER_DECL cv::Mat VisCURVEOcr(const cv::Mat &im,
                                    const OCRCURVEResult &ocr_result,
                                    const float score_threshold = 0);

ULTRAINFER_DECL cv::Mat VisMOT(const cv::Mat &img, const MOTResult &results,
                               float score_threshold = 0.0f,
                               tracking::TrailRecorder *recorder = nullptr);
/** \brief Swap the image background with MattingResult
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] background the background image data,
 * comes from cv::imread(), is a 3-D array with layout HWC, BGR format
 * \param[in] result the MattingResult produced by model
 * \param[in] remove_small_connected_area if remove_small_connected_area==true,
 * the visualized result will not include the small connected areas \return
 * cv::Mat type stores the visualized results
 */
ULTRAINFER_DECL cv::Mat
SwapBackground(const cv::Mat &im, const cv::Mat &background,
               const MattingResult &result,
               bool remove_small_connected_area = false);
/** \brief Swap the image background with SegmentationResult
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] background the background image data,
 * comes from cv::imread(), is a 3-D array with layout HWC, BGR format
 * \param[in] result the SegmentationResult produced by model
 * \param[in] background_label the background label number in SegmentationResult
 * \return cv::Mat type stores the visualized results
 */
ULTRAINFER_DECL cv::Mat SwapBackground(const cv::Mat &im,
                                       const cv::Mat &background,
                                       const SegmentationResult &result,
                                       int background_label);

/** \brief Show the visualized results for key point detection models
 *
 * \param[in] im the input image data, comes from cv::imread(), is a 3-D array
 * with layout HWC, BGR format \param[in] results the result produced by model
 * \param[in] conf_threshold threshold for result scores, the result will not be
 * shown if the score is less than conf_threshold \return cv::Mat type stores
 * the visualized results
 */
ULTRAINFER_DECL cv::Mat
VisKeypointDetection(const cv::Mat &im, const KeyPointDetectionResult &results,
                     float conf_threshold = 0.5f);
ULTRAINFER_DECL cv::Mat VisHeadPose(const cv::Mat &im,
                                    const HeadPoseResult &result, int size = 50,
                                    int line_size = 1);

} // namespace vision
} // namespace ultra_infer
