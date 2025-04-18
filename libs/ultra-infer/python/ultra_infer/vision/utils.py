# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
import json
from .. import c_lib_wrap as C


def mask_to_json(result):
    r_json = {
        "data": result.data,
        "shape": result.shape,
    }
    return json.dumps(r_json)


def detection_to_json(result):
    masks = []
    for mask in result.masks:
        masks.append(mask_to_json(mask))
    r_json = {
        "boxes": result.boxes,
        "scores": result.scores,
        "label_ids": result.label_ids,
        "masks": masks,
        "contain_masks": result.contain_masks,
    }
    return json.dumps(r_json)


def perception_to_json(result):
    r_json = {
        "scores": result.scores,
        "label_ids": result.label_ids,
        "boxes": result.boxes,
        "center": result.center,
        "observation_angle": result.observation_angle,
        "yaw_angle": result.yaw_angle,
        "velocity": result.velocity,
    }
    return json.dumps(r_json)


def classify_to_json(result):
    r_json = {
        "label_ids": result.label_ids,
        "scores": result.scores,
    }
    return json.dumps(r_json)


def keypoint_to_json(result):
    r_json = {
        "keypoints": result.keypoints,
        "scores": result.scores,
        "num_joints": result.num_joints,
    }
    return json.dumps(r_json)


def ocr_to_json(result):
    r_json = {
        "boxes": result.boxes,
        "text": result.text,
        "rec_scores": result.rec_scores,
        "cls_scores": result.cls_scores,
        "cls_labels": result.cls_labels,
    }
    return json.dumps(r_json)


def mot_to_json(result):
    r_json = {
        "boxes": result.boxes,
        "ids": result.ids,
        "scores": result.scores,
        "class_ids": result.class_ids,
    }
    return json.dumps(r_json)


def face_detection_to_json(result):
    r_json = {
        "boxes": result.boxes,
        "landmarks": result.landmarks,
        "scores": result.scores,
        "landmarks_per_face": result.landmarks_per_face,
    }
    return json.dumps(r_json)


def face_alignment_to_json(result):
    r_json = {
        "landmarks": result.landmarks,
    }
    return json.dumps(r_json)


def face_recognition_to_json(result):
    r_json = {
        "embedding": result.embedding,
    }
    return json.dumps(r_json)


def segmentation_to_json(result):
    r_json = {
        "label_map": result.label_map,
        "score_map": result.score_map,
        "shape": result.shape,
        "contain_score_map": result.contain_score_map,
    }
    return json.dumps(r_json)


def matting_to_json(result):
    r_json = {
        "alpha": result.alpha,
        "foreground": result.foreground,
        "shape": result.shape,
        "contain_foreground": result.contain_foreground,
    }
    return json.dumps(r_json)


def head_pose_to_json(result):
    r_json = {
        "euler_angles": result.euler_angles,
    }
    return json.dumps(r_json)


def fd_result_to_json(result):
    if isinstance(result, list):
        r_list = []
        for r in result:
            r_list.append(fd_result_to_json(r))
        return r_list
    elif isinstance(result, C.vision.DetectionResult):
        return detection_to_json(result)
    elif isinstance(result, C.vision.Mask):
        return mask_to_json(result)
    elif isinstance(result, C.vision.ClassifyResult):
        return classify_to_json(result)
    elif isinstance(result, C.vision.KeyPointDetectionResult):
        return keypoint_to_json(result)
    elif isinstance(result, C.vision.OCRResult):
        return ocr_to_json(result)
    elif isinstance(result, C.vision.MOTResult):
        return mot_to_json(result)
    elif isinstance(result, C.vision.FaceDetectionResult):
        return face_detection_to_json(result)
    elif isinstance(result, C.vision.FaceAlignmentResult):
        return face_alignment_to_json(result)
    elif isinstance(result, C.vision.FaceRecognitionResult):
        return face_recognition_to_json(result)
    elif isinstance(result, C.vision.SegmentationResult):
        return segmentation_to_json(result)
    elif isinstance(result, C.vision.MattingResult):
        return matting_to_json(result)
    elif isinstance(result, C.vision.HeadPoseResult):
        return head_pose_to_json(result)
    elif isinstance(result, C.vision.PerceptionResult):
        return perception_to_json(result)
    else:
        assert False, "{} Conversion to JSON format is not supported".format(
            type(result)
        )
    return {}


def json_to_mask(result):
    mask = C.vision.Mask()
    mask.data = result["data"]
    mask.shape = result["shape"]
    return mask


def json_to_detection(result):
    masks = []
    for mask in result["masks"]:
        masks.append(json_to_mask(json.loads(mask)))
    det_result = C.vision.DetectionResult()
    det_result.boxes = result["boxes"]
    det_result.scores = result["scores"]
    det_result.label_ids = result["label_ids"]
    det_result.masks = masks
    det_result.contain_masks = result["contain_masks"]
    return det_result


def json_to_perception(result):
    perception_result = C.vision.PerceptionResult()
    perception_result.scores = result["scores"]
    perception_result.label_ids = result["label_ids"]
    perception_result.boxes = result["boxes"]
    perception_result.center = result["center"]
    perception_result.observation_angle = result["observation_angle"]
    perception_result.yaw_angle = result["yaw_angle"]
    perception_result.velocity = result["velocity"]
    return perception_result


def json_to_classify(result):
    cls_result = C.vision.ClassifyResult()
    cls_result.label_ids = result["label_ids"]
    cls_result.scores = result["scores"]
    return cls_result


def json_to_keypoint(result):
    kp_result = C.vision.KeyPointDetectionResult()
    kp_result.keypoints = result["keypoints"]
    kp_result.scores = result["scores"]
    kp_result.num_joints = result["num_joints"]
    return kp_result


def json_to_ocr(result):
    ocr_result = C.vision.OCRResult()
    ocr_result.boxes = result["boxes"]
    ocr_result.text = result["text"]
    ocr_result.rec_scores = result["rec_scores"]
    ocr_result.cls_scores = result["cls_scores"]
    ocr_result.cls_labels = result["cls_labels"]
    return ocr_result


def json_to_mot(result):
    mot_result = C.vision.MOTResult()
    mot_result.boxes = result["boxes"]
    mot_result.ids = result["ids"]
    mot_result.scores = result["scores"]
    mot_result.class_ids = result["class_ids"]
    return mot_result


def json_to_face_detection(result):
    face_result = C.vision.FaceDetectionResult()
    face_result.boxes = result["boxes"]
    face_result.landmarks = result["landmarks"]
    face_result.scores = result["scores"]
    face_result.landmarks_per_face = result["landmarks_per_face"]
    return face_result


def json_to_face_alignment(result):
    face_result = C.vision.FaceAlignmentResult()
    face_result.landmarks = result["landmarks"]
    return face_result


def json_to_face_recognition(result):
    face_result = C.vision.FaceRecognitionResult()
    face_result.embedding = result["embedding"]
    return face_result


def json_to_segmentation(result):
    seg_result = C.vision.SegmentationResult()
    seg_result.label_map = result["label_map"]
    seg_result.score_map = result["score_map"]
    seg_result.shape = result["shape"]
    seg_result.contain_score_map = result["contain_score_map"]
    return seg_result


def json_to_matting(result):
    matting_result = C.vision.MattingResult()
    matting_result.alpha = result["alpha"]
    matting_result.foreground = result["foreground"]
    matting_result.shape = result["shape"]
    matting_result.contain_foreground = result["contain_foreground"]
    return matting_result


def json_to_head_pose(result):
    hp_result = C.vision.HeadPoseResult()
    hp_result.euler_angles = result["euler_angles"]
    return hp_result
