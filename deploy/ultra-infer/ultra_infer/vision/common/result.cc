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
#include "ultra_infer/vision/common/result.h"

namespace ultra_infer {
namespace vision {

void ClassifyResult::Free() {
  std::vector<int32_t>().swap(label_ids);
  std::vector<float>().swap(scores);
  std::vector<float>().swap(feature);
}

void ClassifyResult::Clear() {
  label_ids.clear();
  scores.clear();
  feature.clear();
}

void ClassifyResult::Resize(int size) {
  label_ids.resize(size);
  scores.resize(size);
  // TODO(qiuyanjun): feature not perform resize now.
  // may need the code below for future.
  // feature.resize(size);
}

std::string ClassifyResult::Str() {
  std::string out;
  out = "ClassifyResult(\nlabel_ids: ";
  for (size_t i = 0; i < label_ids.size(); ++i) {
    out = out + std::to_string(label_ids[i]) + ", ";
  }
  out += "\nscores: ";
  for (size_t i = 0; i < scores.size(); ++i) {
    out = out + std::to_string(scores[i]) + ", ";
  }
  if (!feature.empty()) {
    out += "\nfeature: size (";
    out += std::to_string(feature.size()) + "), only show first 100 values.\n";
    for (size_t i = 0; i < feature.size(); ++i) {
      // only show first 100 values.
      if ((i + 1) <= 100) {
        out = out + std::to_string(feature[i]) + ", ";
        if ((i + 1) % 10 == 0 && (i + 1) < 100) {
          out += "\n";
        }
        if ((i + 1) == 100) {
          out += "\n......";
        }
      }
    }
  }
  out += "\n)";
  return out;
}

ClassifyResult &ClassifyResult::operator=(ClassifyResult &&other) {
  if (&other != this) {
    label_ids = std::move(other.label_ids);
    scores = std::move(other.scores);
    feature = std::move(other.feature);
  }
  return *this;
}

void Mask::Reserve(int size) { data.reserve(size); }

void Mask::Resize(int size) { data.resize(size); }

void Mask::Free() {
  std::vector<uint32_t>().swap(data);
  std::vector<int64_t>().swap(shape);
}

void Mask::Clear() {
  data.clear();
  shape.clear();
}

std::string Mask::Str() {
  std::string out = "Mask(";
  size_t ndim = shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    if (i < ndim - 1) {
      out += std::to_string(shape[i]) + ",";
    } else {
      out += std::to_string(shape[i]);
    }
  }
  out += ")\n";
  return out;
}

DetectionResult::DetectionResult(const DetectionResult &res) {
  boxes.assign(res.boxes.begin(), res.boxes.end());
  rotated_boxes.assign(res.rotated_boxes.begin(), res.rotated_boxes.end());
  scores.assign(res.scores.begin(), res.scores.end());
  label_ids.assign(res.label_ids.begin(), res.label_ids.end());
  contain_masks = res.contain_masks;
  if (contain_masks) {
    masks.clear();
    size_t mask_size = res.masks.size();
    for (size_t i = 0; i < mask_size; ++i) {
      masks.emplace_back(res.masks[i]);
    }
  }
}

DetectionResult &DetectionResult::operator=(DetectionResult &&other) {
  if (&other != this) {
    boxes = std::move(other.boxes);
    rotated_boxes = std::move(other.rotated_boxes);
    scores = std::move(other.scores);
    label_ids = std::move(other.label_ids);
    contain_masks = std::move(other.contain_masks);
    if (contain_masks) {
      masks.clear();
      masks = std::move(other.masks);
    }
  }
  return *this;
}

void DetectionResult::Free() {
  std::vector<std::array<float, 4>>().swap(boxes);
  std::vector<std::array<float, 8>>().swap(rotated_boxes);
  std::vector<float>().swap(scores);
  std::vector<int32_t>().swap(label_ids);
  std::vector<Mask>().swap(masks);
  contain_masks = false;
}

void DetectionResult::Clear() {
  boxes.clear();
  rotated_boxes.clear();
  scores.clear();
  label_ids.clear();
  masks.clear();
  contain_masks = false;
}

void DetectionResult::Reserve(int size) {
  boxes.reserve(size);
  rotated_boxes.reserve(size);
  scores.reserve(size);
  label_ids.reserve(size);
  if (contain_masks) {
    masks.reserve(size);
  }
}

void DetectionResult::Resize(int size) {
  boxes.resize(size);
  rotated_boxes.resize(size);
  scores.resize(size);
  label_ids.resize(size);
  if (contain_masks) {
    masks.resize(size);
  }
}

std::string DetectionResult::Str() {
  std::string out;
  if (!contain_masks) {
    out = "DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]\n";
    if (!rotated_boxes.empty()) {
      out = "DetectionResult: [x1, y1, x2, y2, x3, y3, x4, y4, score, "
            "label_id]\n";
    }
  } else {
    out = "DetectionResult: [xmin, ymin, xmax, ymax, score, label_id, "
          "mask_shape]\n";
    if (!rotated_boxes.empty()) {
      out =
          "DetectionResult: [x1, y1, x2, y2, x3, y3, x4, y4, score, label_id, "
          "mask_shape]\n";
    }
  }
  for (size_t i = 0; i < boxes.size(); ++i) {
    out = out + std::to_string(boxes[i][0]) + "," +
          std::to_string(boxes[i][1]) + ", " + std::to_string(boxes[i][2]) +
          ", " + std::to_string(boxes[i][3]) + ", " +
          std::to_string(scores[i]) + ", " + std::to_string(label_ids[i]);
    if (!contain_masks) {
      out += "\n";
    } else {
      out += ", " + masks[i].Str();
    }
  }

  for (size_t i = 0; i < rotated_boxes.size(); ++i) {
    out = out + std::to_string(rotated_boxes[i][0]) + "," +
          std::to_string(rotated_boxes[i][1]) + ", " +
          std::to_string(rotated_boxes[i][2]) + ", " +
          std::to_string(rotated_boxes[i][3]) + ", " +
          std::to_string(rotated_boxes[i][4]) + "," +
          std::to_string(rotated_boxes[i][5]) + ", " +
          std::to_string(rotated_boxes[i][6]) + ", " +
          std::to_string(rotated_boxes[i][7]) + ", " +
          std::to_string(scores[i]) + ", " + std::to_string(label_ids[i]);
    out += "\n";
  }
  return out;
}

// PerceptionResult -----------------------------------------------------
PerceptionResult::PerceptionResult(const PerceptionResult &res) {
  scores.assign(res.scores.begin(), res.scores.end());
  label_ids.assign(res.label_ids.begin(), res.label_ids.end());
  boxes.assign(res.boxes.begin(), res.boxes.end());
  center.assign(res.center.begin(), res.center.end());
  observation_angle.assign(res.observation_angle.begin(),
                           res.observation_angle.end());
  yaw_angle.assign(res.yaw_angle.begin(), res.yaw_angle.end());
  velocity.assign(res.velocity.begin(), res.velocity.end());
  valid.assign(res.valid.begin(), res.valid.end());
}

PerceptionResult &PerceptionResult::operator=(PerceptionResult &&other) {
  if (&other != this) {
    scores = std::move(other.scores);
    label_ids = std::move(other.label_ids);
    boxes = std::move(other.boxes);
    center = std::move(other.center);
    observation_angle = std::move(other.observation_angle);
    yaw_angle = std::move(other.yaw_angle);
    velocity = std::move(other.velocity);
    valid = std::move(other.valid);
  }
  return *this;
}

void PerceptionResult::Free() {
  std::vector<float>().swap(scores);
  std::vector<int32_t>().swap(label_ids);
  std::vector<std::array<float, 7>>().swap(boxes);
  std::vector<std::array<float, 3>>().swap(center);
  std::vector<float>().swap(observation_angle);
  std::vector<float>().swap(yaw_angle);
  std::vector<std::array<float, 3>>().swap(velocity);
  std::vector<bool>().swap(valid);
}

void PerceptionResult::Clear() {
  scores.clear();
  label_ids.clear();
  boxes.clear();
  center.clear();
  observation_angle.clear();
  yaw_angle.clear();
  velocity.clear();
  valid.clear();
}

void PerceptionResult::Reserve(int size) {
  scores.reserve(size);
  label_ids.reserve(size);
  boxes.reserve(size);
  center.reserve(size);
  observation_angle.reserve(size);
  yaw_angle.reserve(size);
  velocity.reserve(size);
}

void PerceptionResult::Resize(int size) {
  scores.resize(size);
  label_ids.resize(size);
  boxes.resize(size);
  center.resize(size);
  observation_angle.resize(size);
  yaw_angle.resize(size);
  velocity.resize(size);
}

std::string PerceptionResult::Str() {
  std::string out;
  out = "PerceptionResult: [";
  if (valid[2]) {
    out += "xmin, ymin, xmax, ymax, w, h, l,";
  }
  if (valid[3]) {
    out += " cx, cy, cz,";
  }
  if (valid[5]) {
    out += " yaw_angle,";
  }
  if (valid[4]) {
    out += " ob_angle,";
  }
  if (valid[0]) {
    out += " score,";
  }
  if (valid[1]) {
    out += " label_id,";
  }
  out += "]\n";

  for (size_t i = 0; i < boxes.size(); ++i) {
    if (valid[2]) {
      out = out + std::to_string(boxes[i][0]) + "," +
            std::to_string(boxes[i][1]) + ", " + std::to_string(boxes[i][2]) +
            ", " + std::to_string(boxes[i][3]) + ", " +
            std::to_string(boxes[i][4]) + ", " + std::to_string(boxes[i][5]) +
            ", " + std::to_string(boxes[i][6]) + ", ";
    }
    if (valid[3]) {
      out = out + std::to_string(center[i][0]) + ", " +
            std::to_string(center[i][1]) + ", " + std::to_string(center[i][2]) +
            ", ";
    }
    if (valid[5]) {
      out = out + std::to_string(yaw_angle[i]) + ", ";
    }
    if (valid[4]) {
      out = out + std::to_string(observation_angle[i]) + ", ";
    }
    if (valid[0]) {
      out = out + std::to_string(scores[i]) + ", ";
    }
    if (valid[1]) {
      out = out + std::to_string(label_ids[i]);
    }
    out += "\n";
  }
  return out;
}

// PerceptionResult finished

void KeyPointDetectionResult::Free() {
  std::vector<std::array<float, 2>>().swap(keypoints);
  std::vector<float>().swap(scores);
  num_joints = -1;
}

void KeyPointDetectionResult::Clear() {
  keypoints.clear();
  scores.clear();
  num_joints = -1;
}

void KeyPointDetectionResult::Reserve(int size) { keypoints.reserve(size); }

void KeyPointDetectionResult::Resize(int size) { keypoints.resize(size); }

std::string KeyPointDetectionResult::Str() {
  std::string out;

  out = "KeyPointDetectionResult: [x, y, conf]\n";
  for (size_t i = 0; i < keypoints.size(); ++i) {
    out = out + std::to_string(keypoints[i][0]) + "," +
          std::to_string(keypoints[i][1]) + ", " + std::to_string(scores[i]) +
          "\n";
  }
  out += "num_joints:" + std::to_string(num_joints) + "\n";
  return out;
}

void OCRResult::Clear() {
  boxes.clear();
  text.clear();
  rec_scores.clear();
  cls_scores.clear();
  cls_labels.clear();
}

void OCRCURVEResult::Clear() {
  boxes.clear();
  text.clear();
  rec_scores.clear();
  cls_scores.clear();
  cls_labels.clear();
}

void MOTResult::Clear() {
  boxes.clear();
  ids.clear();
  scores.clear();
  class_ids.clear();
}

std::string MOTResult::Str() {
  std::string out;
  out = "MOTResult:\nall boxes counts: " + std::to_string(boxes.size()) + "\n";
  out += "[xmin\tymin\txmax\tymax\tid\tscore]\n";
  for (size_t i = 0; i < boxes.size(); ++i) {
    out = out + "[" + std::to_string(boxes[i][0]) + "\t" +
          std::to_string(boxes[i][1]) + "\t" + std::to_string(boxes[i][2]) +
          "\t" + std::to_string(boxes[i][3]) + "\t" + std::to_string(ids[i]) +
          "\t" + std::to_string(scores[i]) + "]\n";
  }
  return out;
}

FaceDetectionResult::FaceDetectionResult(const FaceDetectionResult &res) {
  boxes.assign(res.boxes.begin(), res.boxes.end());
  landmarks.assign(res.landmarks.begin(), res.landmarks.end());
  scores.assign(res.scores.begin(), res.scores.end());
  landmarks_per_face = res.landmarks_per_face;
}

void FaceDetectionResult::Free() {
  std::vector<std::array<float, 4>>().swap(boxes);
  std::vector<float>().swap(scores);
  std::vector<std::array<float, 2>>().swap(landmarks);
  landmarks_per_face = 0;
}

void FaceDetectionResult::Clear() {
  boxes.clear();
  scores.clear();
  landmarks.clear();
  landmarks_per_face = 0;
}

void FaceDetectionResult::Reserve(int size) {
  boxes.reserve(size);
  scores.reserve(size);
  if (landmarks_per_face > 0) {
    landmarks.reserve(size * landmarks_per_face);
  }
}

void FaceDetectionResult::Resize(int size) {
  boxes.resize(size);
  scores.resize(size);
  if (landmarks_per_face > 0) {
    landmarks.resize(size * landmarks_per_face);
  }
}

std::string FaceDetectionResult::Str() {
  std::string out;
  // format without landmarks
  if (landmarks_per_face <= 0) {
    out = "FaceDetectionResult: [xmin, ymin, xmax, ymax, score]\n";
    for (size_t i = 0; i < boxes.size(); ++i) {
      out = out + std::to_string(boxes[i][0]) + "," +
            std::to_string(boxes[i][1]) + ", " + std::to_string(boxes[i][2]) +
            ", " + std::to_string(boxes[i][3]) + ", " +
            std::to_string(scores[i]) + "\n";
    }
    return out;
  }
  // format with landmarks
  FDASSERT((landmarks.size() == boxes.size() * landmarks_per_face),
           "The size of landmarks != boxes.size * landmarks_per_face.");
  out = "FaceDetectionResult: [xmin, ymin, xmax, ymax, score, (x, y) x " +
        std::to_string(landmarks_per_face) + "]\n";
  for (size_t i = 0; i < boxes.size(); ++i) {
    out = out + std::to_string(boxes[i][0]) + "," +
          std::to_string(boxes[i][1]) + ", " + std::to_string(boxes[i][2]) +
          ", " + std::to_string(boxes[i][3]) + ", " +
          std::to_string(scores[i]) + ", ";
    for (size_t j = 0; j < landmarks_per_face; ++j) {
      out = out + "(" +
            std::to_string(landmarks[i * landmarks_per_face + j][0]) + "," +
            std::to_string(landmarks[i * landmarks_per_face + j][1]);
      if (j < landmarks_per_face - 1) {
        out = out + "), ";
      } else {
        out = out + ")\n";
      }
    }
  }
  return out;
}

void FaceAlignmentResult::Free() {
  std::vector<std::array<float, 2>>().swap(landmarks);
}

void FaceAlignmentResult::Clear() { landmarks.clear(); }

void FaceAlignmentResult::Reserve(int size) { landmarks.resize(size); }

void FaceAlignmentResult::Resize(int size) { landmarks.resize(size); }

std::string FaceAlignmentResult::Str() {
  std::string out;

  out = "FaceAlignmentResult: [x, y]\n";
  out = out + "There are " + std::to_string(landmarks.size()) +
        " landmarks, the top 10 are listed as below:\n";
  int landmarks_size = landmarks.size();
  size_t result_length = std::min(10, landmarks_size);
  for (size_t i = 0; i < result_length; ++i) {
    out = out + std::to_string(landmarks[i][0]) + "," +
          std::to_string(landmarks[i][1]) + "\n";
  }
  out += "num_landmarks:" + std::to_string(landmarks.size()) + "\n";
  return out;
}

void SegmentationResult::Clear() {
  label_map.clear();
  score_map.clear();
  shape.clear();
  contain_score_map = false;
}

void SegmentationResult::Free() {
  std::vector<uint8_t>().swap(label_map);
  std::vector<float>().swap(score_map);
  std::vector<int64_t>().swap(shape);
  contain_score_map = false;
}

void SegmentationResult::Reserve(int size) {
  label_map.reserve(size);
  if (contain_score_map) {
    score_map.reserve(size);
  }
}

void SegmentationResult::Resize(int size) {
  label_map.resize(size);
  if (contain_score_map) {
    score_map.resize(size);
  }
}

std::string SegmentationResult::Str() {
  std::string out;
  out = "SegmentationResult Image masks 10 rows x 10 cols: \n";
  for (size_t i = 0; i < 10; ++i) {
    out += "[";
    for (size_t j = 0; j < 10; ++j) {
      out = out + std::to_string(label_map[i * 10 + j]) + ", ";
    }
    out += ".....]\n";
  }
  out += "...........\n";
  if (contain_score_map) {
    out += "SegmentationResult Score map 10 rows x 10 cols: \n";
    for (size_t i = 0; i < 10; ++i) {
      out += "[";
      for (size_t j = 0; j < 10; ++j) {
        out = out + std::to_string(score_map[i * 10 + j]) + ", ";
      }
      out += ".....]\n";
    }
    out += "...........\n";
  }
  out += "result shape is: [" + std::to_string(shape[0]) + " " +
         std::to_string(shape[1]) + "]";
  return out;
}

SegmentationResult &SegmentationResult::operator=(SegmentationResult &&other) {
  if (&other != this) {
    label_map = std::move(other.label_map);
    shape = std::move(other.shape);
    contain_score_map = std::move(other.contain_score_map);
    if (contain_score_map) {
      score_map.clear();
      score_map = std::move(other.score_map);
    }
  }
  return *this;
}
FaceRecognitionResult::FaceRecognitionResult(const FaceRecognitionResult &res) {
  embedding.assign(res.embedding.begin(), res.embedding.end());
}

void FaceRecognitionResult::Free() { std::vector<float>().swap(embedding); }

void FaceRecognitionResult::Clear() { embedding.clear(); }

void FaceRecognitionResult::Reserve(int size) { embedding.reserve(size); }

void FaceRecognitionResult::Resize(int size) { embedding.resize(size); }

std::string FaceRecognitionResult::Str() {
  std::string out;
  out = "FaceRecognitionResult: [";
  size_t numel = embedding.size();
  if (numel <= 0) {
    return out + "Empty Result]";
  }
  // max, min, mean
  float min_val = embedding.at(0);
  float max_val = embedding.at(0);
  float total_val = embedding.at(0);
  for (size_t i = 1; i < numel; ++i) {
    float val = embedding.at(i);
    total_val += val;
    if (val < min_val) {
      min_val = val;
    }
    if (val > max_val) {
      max_val = val;
    }
  }
  float mean_val = total_val / static_cast<float>(numel);
  out = out + "Dim(" + std::to_string(numel) + "), " + "Min(" +
        std::to_string(min_val) + "), " + "Max(" + std::to_string(max_val) +
        "), " + "Mean(" + std::to_string(mean_val) + ")]\n";
  return out;
}

MattingResult::MattingResult(const MattingResult &res) {
  alpha.assign(res.alpha.begin(), res.alpha.end());
  foreground.assign(res.foreground.begin(), res.foreground.end());
  shape.assign(res.shape.begin(), res.shape.end());
  contain_foreground = res.contain_foreground;
}

void MattingResult::Clear() {
  alpha.clear();
  foreground.clear();
  shape.clear();
  contain_foreground = false;
}

void MattingResult::Free() {
  std::vector<float>().swap(alpha);
  std::vector<float>().swap(foreground);
  std::vector<int64_t>().swap(shape);
  contain_foreground = false;
}

void MattingResult::Reserve(int size) {
  alpha.reserve(size);
  if (contain_foreground) {
    FDASSERT((shape.size() == 3),
             "Please initial shape (h,w,c) before call Reserve.");
    int c = static_cast<int>(shape[2]);
    foreground.reserve(size * c);
  }
}

void MattingResult::Resize(int size) {
  alpha.resize(size);
  if (contain_foreground) {
    FDASSERT((shape.size() == 3),
             "Please initial shape (h,w,c) before call Resize.");
    int c = static_cast<int>(shape[2]);
    foreground.resize(size * c);
  }
}

std::string MattingResult::Str() {
  std::string out;
  out = "MattingResult[";
  if (contain_foreground) {
    out += "Foreground(true)";
  } else {
    out += "Foreground(false)";
  }
  out += ", Alpha(";
  size_t numel = alpha.size();
  if (numel <= 0) {
    return out + "[Empty Result]";
  }
  // max, min, mean
  float min_val = alpha.at(0);
  float max_val = alpha.at(0);
  float total_val = alpha.at(0);
  for (size_t i = 1; i < numel; ++i) {
    float val = alpha.at(i);
    total_val += val;
    if (val < min_val) {
      min_val = val;
    }
    if (val > max_val) {
      max_val = val;
    }
  }
  float mean_val = total_val / static_cast<float>(numel);
  // shape
  std::string shape_str = "Shape(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if ((i + 1) != shape.size()) {
      shape_str += std::to_string(shape[i]) + ",";
    } else {
      shape_str += std::to_string(shape[i]) + ")";
    }
  }
  out = out + "Numel(" + std::to_string(numel) + "), " + shape_str + ", Min(" +
        std::to_string(min_val) + "), " + "Max(" + std::to_string(max_val) +
        "), " + "Mean(" + std::to_string(mean_val) + "))]\n";
  return out;
}

std::string OCRResult::Str() {
  std::string no_result;
  if (boxes.size() > 0) {
    std::string out;
    for (int n = 0; n < boxes.size(); n++) {
      out = out + "det boxes: [";
      for (int i = 0; i < 4; i++) {
        out = out + "[" + std::to_string(boxes[n][i * 2]) + "," +
              std::to_string(boxes[n][i * 2 + 1]) + "]";

        if (i != 3) {
          out = out + ",";
        }
      }
      out = out + "]";

      if (rec_scores.size() > 0) {
        out = out + "rec text: " + text[n] +
              " rec score:" + std::to_string(rec_scores[n]) + " ";
      }
      if (cls_labels.size() > 0) {
        out = out + "cls label: " + std::to_string(cls_labels[n]) +
              " cls score: " + std::to_string(cls_scores[n]);
      }
      out = out + "\n";
    }

    if (table_boxes.size() > 0 && table_structure.size() > 0) {
      for (int n = 0; n < boxes.size(); n++) {
        out = out + "table boxes: [";
        for (int i = 0; i < 4; i++) {
          out = out + "[" + std::to_string(table_boxes[n][i * 2]) + "," +
                std::to_string(table_boxes[n][i * 2 + 1]) + "]";

          if (i != 3) {
            out = out + ",";
          }
        }
        out = out + "]\n";
      }

      out = out + "\ntable structure: \n";
      for (int m = 0; m < table_structure.size(); m++) {
        out += table_structure[m];
      }

      if (!table_html.empty()) {
        out = out + "\n" + "table html: \n" + table_html;
      }
    }
    std::vector<std::array<int, 8>> table_boxes;
    std::vector<std::string> table_structure;
    return out;

  } else if (boxes.size() == 0 && rec_scores.size() > 0 &&
             cls_scores.size() > 0) {
    std::string out;
    for (int i = 0; i < rec_scores.size(); i++) {
      out = out + "rec text: " + text[i] +
            " rec score:" + std::to_string(rec_scores[i]) + " ";
      out = out + "cls label: " + std::to_string(cls_labels[i]) +
            " cls score: " + std::to_string(cls_scores[i]);
      out = out + "\n";
    }
    return out;
  } else if (boxes.size() == 0 && rec_scores.size() == 0 &&
             cls_scores.size() > 0) {
    std::string out;
    for (int i = 0; i < cls_scores.size(); i++) {
      out = out + "cls label: " + std::to_string(cls_labels[i]) +
            " cls score: " + std::to_string(cls_scores[i]);
      out = out + "\n";
    }
    return out;
  } else if (boxes.size() == 0 && rec_scores.size() > 0 &&
             cls_scores.size() == 0) {
    std::string out;
    for (int i = 0; i < rec_scores.size(); i++) {
      out = out + "rec text: " + text[i] +
            " rec score:" + std::to_string(rec_scores[i]) + " ";
      out = out + "\n";
    }
    return out;
  } else if (boxes.size() == 0 && table_boxes.size() > 0 &&
             table_structure.size() > 0) {
    std::string out;
    for (int n = 0; n < table_boxes.size(); n++) {
      out = out + "table boxes: [";
      for (int i = 0; i < 4; i++) {
        out = out + "[" + std::to_string(table_boxes[n][i * 2]) + "," +
              std::to_string(table_boxes[n][i * 2 + 1]) + "]";

        if (i != 3) {
          out = out + ",";
        }
      }
      out = out + "]\n";
    }

    out = out + "\ntable structure: \n";
    for (int m = 0; m < table_structure.size(); m++) {
      out += table_structure[m];
    }

    if (!table_html.empty()) {
      out = out + "\n" + "table html: \n" + table_html;
    }
    return out;
  }

  no_result = no_result + "No Results!";
  return no_result;
}

std::string OCRCURVEResult::Str() {
  std::string no_result;
  if (boxes.size() > 0) {
    std::string out;
    for (int n = 0; n < boxes.size(); n++) {
      out = out + "det boxes: [";
      for (int i = 0; i < boxes[n].size() / 2; i++) {
        out = out + "[" + std::to_string(boxes[n][i * 2]) + "," +
              std::to_string(boxes[n][i * 2 + 1]) + "]";

        if (i != boxes[n].size() / 2 - 1) {
          out = out + ",";
        }
      }
      out = out + "]";

      if (rec_scores.size() > 0) {
        out = out + "rec text: " + text[n] +
              " rec score:" + std::to_string(rec_scores[n]) + " ";
      }
      if (cls_labels.size() > 0) {
        out = out + "cls label: " + std::to_string(cls_labels[n]) +
              " cls score: " + std::to_string(cls_scores[n]);
      }
      out = out + "\n";
    }

    if (table_boxes.size() > 0 && table_structure.size() > 0) {
      for (int n = 0; n < boxes.size(); n++) {
        out = out + "table boxes: [";
        for (int i = 0; i < 4; i++) {
          out = out + "[" + std::to_string(table_boxes[n][i * 2]) + "," +
                std::to_string(table_boxes[n][i * 2 + 1]) + "]";

          if (i != 3) {
            out = out + ",";
          }
        }
        out = out + "]\n";
      }

      out = out + "\ntable structure: \n";
      for (int m = 0; m < table_structure.size(); m++) {
        out += table_structure[m];
      }

      if (!table_html.empty()) {
        out = out + "\n" + "table html: \n" + table_html;
      }
    }
    std::vector<std::array<int, 8>> table_boxes;
    std::vector<std::string> table_structure;
    return out;

  } else if (boxes.size() == 0 && rec_scores.size() > 0 &&
             cls_scores.size() > 0) {
    std::string out;
    for (int i = 0; i < rec_scores.size(); i++) {
      out = out + "rec text: " + text[i] +
            " rec score:" + std::to_string(rec_scores[i]) + " ";
      out = out + "cls label: " + std::to_string(cls_labels[i]) +
            " cls score: " + std::to_string(cls_scores[i]);
      out = out + "\n";
    }
    return out;
  } else if (boxes.size() == 0 && rec_scores.size() == 0 &&
             cls_scores.size() > 0) {
    std::string out;
    for (int i = 0; i < cls_scores.size(); i++) {
      out = out + "cls label: " + std::to_string(cls_labels[i]) +
            " cls score: " + std::to_string(cls_scores[i]);
      out = out + "\n";
    }
    return out;
  } else if (boxes.size() == 0 && rec_scores.size() > 0 &&
             cls_scores.size() == 0) {
    std::string out;
    for (int i = 0; i < rec_scores.size(); i++) {
      out = out + "rec text: " + text[i] +
            " rec score:" + std::to_string(rec_scores[i]) + " ";
      out = out + "\n";
    }
    return out;
  } else if (boxes.size() == 0 && table_boxes.size() > 0 &&
             table_structure.size() > 0) {
    std::string out;
    for (int n = 0; n < table_boxes.size(); n++) {
      out = out + "table boxes: [";
      for (int i = 0; i < 4; i++) {
        out = out + "[" + std::to_string(table_boxes[n][i * 2]) + "," +
              std::to_string(table_boxes[n][i * 2 + 1]) + "]";

        if (i != 3) {
          out = out + ",";
        }
      }
      out = out + "]\n";
    }

    out = out + "\ntable structure: \n";
    for (int m = 0; m < table_structure.size(); m++) {
      out += table_structure[m];
    }

    if (!table_html.empty()) {
      out = out + "\n" + "table html: \n" + table_html;
    }
    return out;
  }

  no_result = no_result + "No Results!";
  return no_result;
}
void HeadPoseResult::Free() { std::vector<float>().swap(euler_angles); }

void HeadPoseResult::Clear() { euler_angles.clear(); }

void HeadPoseResult::Reserve(int size) { euler_angles.resize(size); }

void HeadPoseResult::Resize(int size) { euler_angles.resize(size); }

std::string HeadPoseResult::Str() {
  std::string out;

  out = "HeadPoseResult: [yaw, pitch, roll]\n";
  out = out + "yaw: " + std::to_string(euler_angles[0]) + "\n" +
        "pitch: " + std::to_string(euler_angles[1]) + "\n" +
        "roll: " + std::to_string(euler_angles[2]) + "\n";
  return out;
}

} // namespace vision
} // namespace ultra_infer
