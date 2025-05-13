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
#include "opencv2/core/core.hpp"
#include "ultra_infer/ultra_infer_model.h"
#include <set>

namespace ultra_infer {
/** \brief All C++ UltraInfer Vision Models APIs are defined inside this
 * namespace
 *
 */
namespace vision {
enum ULTRAINFER_DECL ResultType {
  UNKNOWN_RESULT,
  CLASSIFY,
  DETECTION,
  SEGMENTATION,
  OCR,
  MOT,
  FACE_DETECTION,
  FACE_ALIGNMENT,
  FACE_RECOGNITION,
  MATTING,
  MASK,
  KEYPOINT_DETECTION,
  HEADPOSE,
  PERCEPTION,
};

struct ULTRAINFER_DECL BaseResult {
  ResultType type = ResultType::UNKNOWN_RESULT;
};

/*! @brief Classify result structure for all the image classify models
 */
struct ULTRAINFER_DECL ClassifyResult : public BaseResult {
  ClassifyResult() = default;
  /// Classify result for an image
  std::vector<int32_t> label_ids;
  /// The confidence for each classify result
  std::vector<float> scores;
  /// The feature vector of recognizer, e.g, PP-ShiTuV2 Recognizer
  std::vector<float> feature;
  ResultType type = ResultType::CLASSIFY;

  /// Resize ClassifyResult data buffer
  void Resize(int size);

  /// Clear ClassifyResult
  void Clear();

  /// Clear ClassifyResult and free the memory
  void Free();

  /// Copy constructor
  ClassifyResult(const ClassifyResult &other) = default;
  /// Move assignment
  ClassifyResult &operator=(ClassifyResult &&other);

  /// Debug function, convert the result to string to print
  std::string Str();
};

/*! Mask structure, used in DetectionResult for instance segmentation models
 */
struct ULTRAINFER_DECL Mask : public BaseResult {
  /// Mask data buffer
  std::vector<uint32_t> data;
  /// Shape of mask
  std::vector<int64_t> shape; // (H,W) ...
  ResultType type = ResultType::MASK;

  /// clear Mask result
  void Clear();

  /// Clear Mask result and free the memory
  void Free();

  /// Return a mutable pointer of the mask data buffer
  void *Data() { return data.data(); }

  /// Return a pointer of the mask data buffer for read only
  const void *Data() const { return data.data(); }

  /// Reserve size for mask data buffer
  void Reserve(int size);

  /// Resize the mask data buffer
  void Resize(int size);

  /// Debug function, convert the result to string to print
  std::string Str();
};

/*! @brief Detection result structure for all the object detection models and
 * instance segmentation models
 */
struct ULTRAINFER_DECL DetectionResult : public BaseResult {
  DetectionResult() = default;
  /** \brief All the detected object boxes for an input image, the size of
   * `boxes` is the number of detected objects, and the element of `boxes` is a
   * array of 4 float values, means [xmin, ymin, xmax, ymax]
   */
  std::vector<std::array<float, 4>> boxes;
  /** \brief All the detected rotated object boxes for an input image, the size
   * of `boxes` is the number of detected objects, and the element of
   * `rotated_boxes` is an array of 8 float values, means [x1, y1, x2, y2, x3,
   * y3, x4, y4]
   */
  std::vector<std::array<float, 8>> rotated_boxes;
  /** \brief The confidence for all the detected objects
   */
  std::vector<float> scores;
  /// The classify label for all the detected objects
  std::vector<int32_t> label_ids;
  /** \brief For instance segmentation model, `masks` is the predict mask for
   * all the detected objects
   */
  std::vector<Mask> masks;
  /// Shows if the DetectionResult has mask
  bool contain_masks = false;

  ResultType type = ResultType::DETECTION;

  /// Copy constructor
  DetectionResult(const DetectionResult &res);
  /// Move assignment
  DetectionResult &operator=(DetectionResult &&other);

  /// Clear DetectionResult
  void Clear();

  /// Clear DetectionResult and free the memory
  void Free();

  void Reserve(int size);

  void Resize(int size);

  /// Debug function, convert the result to string to print
  std::string Str();
};

/*! @brief Detection result structure for all the object detection models and
 * instance segmentation models
 */
struct ULTRAINFER_DECL PerceptionResult : public BaseResult {
  PerceptionResult() = default;

  std::vector<float> scores;

  std::vector<int32_t> label_ids;
  // xmin, ymin, xmax, ymax, h, w, l
  std::vector<std::array<float, 7>> boxes;
  // cx, cy, cz
  std::vector<std::array<float, 3>> center;

  std::vector<float> observation_angle;

  std::vector<float> yaw_angle;
  // vx, vy, vz
  std::vector<std::array<float, 3>> velocity;

  // valid results for func Str(): True for printing
  // 0 scores
  // 1 label_ids
  // 2 boxes
  // 3 center
  // 4 observation_angle
  // 5 yaw_angle
  // 6 velocity
  std::vector<bool> valid;

  /// Copy constructor
  PerceptionResult(const PerceptionResult &res);
  /// Move assignment
  PerceptionResult &operator=(PerceptionResult &&other);

  /// Clear PerceptionResult
  void Clear();

  /// Clear PerceptionResult and free the memory
  void Free();

  void Reserve(int size);

  void Resize(int size);

  /// Debug function, convert the result to string to print
  std::string Str();
};

/*! @brief KeyPoint Detection result structure for all the keypoint detection
 * models
 */
struct ULTRAINFER_DECL KeyPointDetectionResult : public BaseResult {
  /** \brief All the coordinates of detected keypoints for an input image, the
   * size of `keypoints` is num_detected_objects * num_joints, and the element
   * of `keypoint` is a array of 2 float values, means [x, y]
   */
  std::vector<std::array<float, 2>> keypoints;
  //// The confidence for all the detected points
  std::vector<float> scores;
  //// Number of joints for a detected object
  int num_joints = -1;

  ResultType type = ResultType::KEYPOINT_DETECTION;
  /// Clear KeyPointDetectionResult
  void Clear();

  /// Clear KeyPointDetectionResult and free the memory
  void Free();

  void Reserve(int size);

  void Resize(int size);

  /// Debug function, convert the result to string to print
  std::string Str();
};

struct ULTRAINFER_DECL OCRResult : public BaseResult {
  std::vector<std::array<int, 8>> boxes;

  std::vector<std::string> text;
  std::vector<float> rec_scores;

  std::vector<float> cls_scores;
  std::vector<int32_t> cls_labels;

  std::vector<std::array<int, 8>> table_boxes;
  std::vector<std::string> table_structure;
  std::string table_html;

  ResultType type = ResultType::OCR;

  void Clear();

  std::string Str();
};

struct ULTRAINFER_DECL OCRCURVEResult : public BaseResult {
  std::vector<std::vector<int>> boxes;
  std::vector<std::string> text;
  std::vector<float> rec_scores;

  std::vector<float> cls_scores;
  std::vector<int32_t> cls_labels;

  std::vector<std::array<int, 8>> table_boxes;
  std::vector<std::string> table_structure;
  std::string table_html;

  ResultType type = ResultType::OCR;

  void Clear();

  std::string Str();
};
/*! @brief MOT(Multi-Object Tracking) result structure for all the MOT models
 */
struct ULTRAINFER_DECL MOTResult : public BaseResult {
  /** \brief All the tracking object boxes for an input image, the size of
   * `boxes` is the number of tracking objects, and the element of `boxes` is a
   * array of 4 float values, means [xmin, ymin, xmax, ymax]
   */
  std::vector<std::array<int, 4>> boxes;
  /** \brief All the tracking object ids
   */
  std::vector<int> ids;
  /** \brief The confidence for all the tracking objects
   */
  std::vector<float> scores;
  /** \brief The classify label id for all the tracking object
   */
  std::vector<int> class_ids;

  ResultType type = ResultType::MOT;
  /// Clear MOT result
  void Clear();
  /// Debug function, convert the result to string to print
  std::string Str();
};

/*! @brief Face detection result structure for all the face detection models
 */
struct ULTRAINFER_DECL FaceDetectionResult : public BaseResult {
  /** \brief All the detected object boxes for an input image, the size of
   * `boxes` is the number of detected objects, and the element of `boxes` is a
   * array of 4 float values, means [xmin, ymin, xmax, ymax]
   */
  std::vector<std::array<float, 4>> boxes;
  /** \brief
   * If the model detect face with landmarks, every detected object box
   * correspoing to a landmark, which is a array of 2 float values, means
   * location [x,y]
   */
  std::vector<std::array<float, 2>> landmarks;
  /** \brief
   * Indicates the confidence of all targets detected from a single image, and
   * the number of elements is consistent with boxes.size()
   */
  std::vector<float> scores;
  ResultType type = ResultType::FACE_DETECTION;
  /** \brief
   * `landmarks_per_face` indicates the number of face landmarks for each
   * detected face if the model's output contains face landmarks (such as
   * YOLOv5Face, SCRFD, ...)
   */
  int landmarks_per_face;

  FaceDetectionResult() { landmarks_per_face = 0; }
  FaceDetectionResult(const FaceDetectionResult &res);
  /// Clear FaceDetectionResult
  void Clear();

  /// Clear FaceDetectionResult and free the memory
  void Free();

  void Reserve(int size);

  void Resize(int size);
  /// Debug function, convert the result to string to print
  std::string Str();
};

/*! @brief Face Alignment result structure for all the face alignment models
 */
struct ULTRAINFER_DECL FaceAlignmentResult : public BaseResult {
  /** \brief All the coordinates of detected landmarks for an input image, and
   * the element of `landmarks` is a array of 2 float values, means [x, y]
   */
  std::vector<std::array<float, 2>> landmarks;

  ResultType type = ResultType::FACE_ALIGNMENT;
  /// Clear FaceAlignmentResult
  void Clear();

  /// Clear FaceAlignmentResult and free the memory
  void Free();

  void Reserve(int size);

  void Resize(int size);

  /// Debug function, convert the result to string to print
  std::string Str();
};

/*! @brief Segmentation result structure for all the segmentation models
 */
struct ULTRAINFER_DECL SegmentationResult : public BaseResult {
  SegmentationResult() = default;
  /** \brief
   * `label_map` stores the pixel-level category labels for input image. the
   * number of pixels is equal to label_map.size()
   */
  std::vector<uint8_t> label_map;
  /** \brief
   * `score_map` stores the probability of the predicted label for each pixel of
   * input image.
   */
  std::vector<float> score_map;
  /// The output shape, means [H, W]
  std::vector<int64_t> shape;
  /// SegmentationResult whether containing score_map
  bool contain_score_map = false;

  /// Copy constructor
  SegmentationResult(const SegmentationResult &other) = default;
  /// Move assignment
  SegmentationResult &operator=(SegmentationResult &&other);

  ResultType type = ResultType::SEGMENTATION;
  /// Clear Segmentation result
  void Clear();

  /// Clear Segmentation result and free the memory
  void Free();

  void Reserve(int size);

  void Resize(int size);

  /// Debug function, convert the result to string to print
  std::string Str();
};

/*! @brief Face recognition result structure for all the Face recognition models
 */
struct ULTRAINFER_DECL FaceRecognitionResult : public BaseResult {
  /** \brief The feature embedding that represents the final extraction of the
   * face recognition model can be used to calculate the feature similarity
   * between faces.
   */
  std::vector<float> embedding;

  ResultType type = ResultType::FACE_RECOGNITION;

  FaceRecognitionResult() {}
  FaceRecognitionResult(const FaceRecognitionResult &res);
  /// Clear FaceRecognitionResult
  void Clear();

  /// Clear FaceRecognitionResult and free the memory
  void Free();

  void Reserve(int size);

  void Resize(int size);
  /// Debug function, convert the result to string to print
  std::string Str();
};

/*! @brief Matting result structure for all the Matting models
 */
struct ULTRAINFER_DECL MattingResult : public BaseResult {
  /** \brief
  `alpha` is a one-dimensional vector, which is the predicted alpha transparency
  value. The range of values is [0., 1.], and the length is hxw. h, w are the
  height and width of the input image
  */
  std::vector<float> alpha; // h x w
  /** \brief
  If the model can predict foreground, `foreground` save the predicted
  foreground image, the shape is [height,width,channel] generally.
  */
  std::vector<float> foreground; // h x w x c (c=3 default)
  /** \brief
   * The shape of output result, when contain_foreground == false, shape only
   * contains (h, w), when contain_foreground == true, shape contains (h, w, c),
   * and c is generally 3
   */
  std::vector<int64_t> shape;
  /** \brief
  If the model can predict alpha matte and foreground, contain_foreground =
  true, default false
  */
  bool contain_foreground = false;

  ResultType type = ResultType::MATTING;

  MattingResult() {}
  MattingResult(const MattingResult &res);
  /// Clear matting result
  void Clear();

  /// Free matting result
  void Free();

  void Reserve(int size);

  void Resize(int size);
  /// Debug function, convert the result to string to print
  std::string Str();
};

/*! @brief HeadPose result structure for all the headpose models
 */
struct ULTRAINFER_DECL HeadPoseResult : public BaseResult {
  /** \brief EulerAngles for an input image, and the element of `euler_angles`
   * is a vector, contains {yaw, pitch, roll}
   */
  std::vector<float> euler_angles;

  ResultType type = ResultType::HEADPOSE;
  /// Clear HeadPoseResult
  void Clear();

  /// Clear HeadPoseResult and free the memory
  void Free();

  void Reserve(int size);

  void Resize(int size);

  /// Debug function, convert the result to string to print
  std::string Str();
};

} // namespace vision
} // namespace ultra_infer
