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
import logging
from ... import c_lib_wrap as C
import cv2


def vis_detection(
    im_data,
    det_result,
    labels=[],
    score_threshold=0.0,
    line_size=1,
    font_size=0.5,
    font_color=[255, 255, 255],
    font_thickness=1,
):
    """Show the visualized results for detection models

    :param im_data: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
    :param det_result: the result produced by model
    :param labels: (list of str) the visualized result will show the bounding box contain class label
    :param score_threshold: (float) score_threshold threshold for result scores, the bounding box will not be shown if the score is less than score_threshold
    :param line_size: (float) line_size line size for bounding boxes
    :param font_size: (float) font_size font size for text
    :param font_color: (list of int) font_color  for text
    :param font_thickness: (int) font_thickness for text
    :return: (numpy.ndarray) image with visualized results
    """
    return C.vision.vis_detection(
        im_data,
        det_result,
        labels,
        score_threshold,
        line_size,
        font_size,
        font_color,
        font_thickness,
    )


def vis_perception(
    im_data, det_result, config_file, score_threshold=0.0, line_size=1, font_size=0.5
):
    """Show the visualized results for 3d detection models

    :param im_data: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
    :param det_result: the result produced by model
    :param config_file: the config file for detection and visualization
    :param score_threshold: (float) score_threshold threshold for result scores, the bounding box will not be shown if the score is less than score_threshold
    :param line_size: (float) line_size line size for bounding boxes
    :param font_size: (float) font_size font size for text
    :return: (numpy.ndarray) image with visualized results
    """
    return C.vision.vis_perception(
        im_data, det_result, config_file, score_threshold, line_size, font_size
    )


def vis_keypoint_detection(im_data, keypoint_det_result, conf_threshold=0.5):
    """Show the visualized results for keypoint detection models

    :param im_data: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
    :param keypoint_det_result: the result produced by model
    :param conf_threshold: (float) conf_threshold threshold for result scores, the bounding box will not be shown if the score is less than conf_threshold
    :return: (numpy.ndarray) image with visualized results
    """
    return C.vision.Visualize.vis_keypoint_detection(
        im_data, keypoint_det_result, conf_threshold
    )


def vis_face_detection(im_data, face_det_result, line_size=1, font_size=0.5):
    """Show the visualized results for face detection models

    :param im_data: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
    :param face_det_result: the result produced by model
    :param line_size: (float) line_size line size for bounding boxes
    :param font_size: (float) font_size font size for text
    :return: (numpy.ndarray) image with visualized results
    """
    return C.vision.vis_face_detection(im_data, face_det_result, line_size, font_size)


def vis_face_alignment(im_data, face_align_result, line_size=1):
    """Show the visualized results for face alignment models

    :param im_data: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
    :param face_align_result: the result produced by model
    :param line_size: (float)line_size line size for circle point
    :return: (numpy.ndarray) image with visualized results
    """
    return C.vision.vis_face_alignment(im_data, face_align_result, line_size)


def vis_segmentation(im_data, seg_result, weight=0.5):
    """Show the visualized results for segmentation models

    :param im_data: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
    :param seg_result: the result produced by model
    :param weight: (float)transparent weight of visualized result image
    :return: (numpy.ndarray) image with visualized results
    """
    return C.vision.vis_segmentation(im_data, seg_result, weight)


def vis_matting_alpha(im_data, matting_result, remove_small_connected_area=False):
    logging.warning(
        "DEPRECATED: ultra_infer.vision.vis_matting_alpha is deprecated, please use ultra_infer.vision.vis_matting function instead."
    )
    return C.vision.vis_matting(im_data, matting_result, remove_small_connected_area)


def vis_matting(
    im_data,
    matting_result,
    transparent_background=False,
    transparent_threshold=0.99,
    remove_small_connected_area=False,
):
    """Show the visualized results for matting models

    :param im_data: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
    :param matting_result: the result produced by model
    :param transparent_background: whether visulizing matting result with transparent background
    :param transparent_threshold: since the alpha value in MattringResult is a float between [0, 1], transparent_threshold is used to filter background pixel
    :param remove_small_connected_area: (bool) if remove_small_connected_area==True, the visualized result will not include the small connected areas
    :return: (numpy.ndarray) image with visualized results
    """
    return C.vision.vis_matting(
        im_data,
        matting_result,
        transparent_background,
        transparent_threshold,
        remove_small_connected_area,
    )


def swap_background_matting(
    im_data, background, result, remove_small_connected_area=False
):
    logging.warning(
        "DEPRECATED: ultra_infer.vision.swap_background_matting is deprecated, please use ultra_infer.vision.swap_background function instead."
    )
    assert isinstance(
        result, C.vision.MattingResult
    ), "The result must be MattingResult type"
    return C.vision.Visualize.swap_background_matting(
        im_data, background, result, remove_small_connected_area
    )


def swap_background_segmentation(im_data, background, background_label, result):
    logging.warning(
        "DEPRECATED: ultra_infer.vision.swap_background_segmentation is deprecated, please use ultra_infer.vision.swap_background function instead."
    )
    assert isinstance(
        result, C.vision.SegmentationResult
    ), "The result must be SegmentaitonResult type"
    return C.vision.Visualize.swap_background_segmentation(
        im_data, background, background_label, result
    )


def swap_background(
    im_data, background, result, remove_small_connected_area=False, background_label=0
):
    """Swap the image background with MattingResult or SegmentationResult

    :param im_data: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
    :param background: (numpy.ndarray)The background image data, 3-D array with layout HWC, BGR format
    :param result: The result produced by model, MattingResult or SegmentationResult
    :param remove_small_connected_area: (bool) If remove_small_connected_area==True, the visualized result will not include the small connected areas
    :param background_label: (int)The background label number in SegmentationResult
    :return: (numpy.ndarray) image with visualized results
    """
    if isinstance(result, C.vision.MattingResult):
        return C.vision.swap_background(
            im_data, background, result, remove_small_connected_area
        )
    elif isinstance(result, C.vision.SegmentationResult):
        return C.vision.swap_background(im_data, background, result, background_label)
    else:
        raise Exception(
            "Only support result type of MattingResult or SegmentationResult, but now the data type is {}.".format(
                type(result)
            )
        )


def vis_ppocr(im_data, det_result):
    """Show the visualized results for ocr models

    :param im_data: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
    :param det_result: the result produced by model
    :return: (numpy.ndarray) image with visualized results
    """
    return C.vision.vis_ppocr(im_data, det_result)


def vis_ppocr_curve(im_data, det_result):
    """Show the visualized results for ocr models

    :param im_data: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
    :param det_result: the result produced by model
    :return: (numpy.ndarray) image with visualized results
    """
    return C.vision.vis_ppocr_curve(im_data, det_result)


def vis_mot(im_data, mot_result, score_threshold=0.0, records=None):
    return C.vision.vis_mot(im_data, mot_result, score_threshold, records)


def vis_headpose(im_data, headpose_result, size=50, line_size=1):
    return C.vision.vis_headpose(im_data, headpose_result, size, line_size)
