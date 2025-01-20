# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import os
from pathlib import Path
from typing import Dict
import copy
import math
import random
import numpy as np
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont
from ....utils.fonts import PINGFANG_FONT_FILE_PATH, create_font
from ...common.result import BaseCVResult, StrMixin, JsonMixin


class OCRResult(BaseCVResult):
    """OCR result"""

    def get_minarea_rect(self, points: np.ndarray) -> np.ndarray:
        """
        Get the minimum area rectangle for the given points using OpenCV.

        Args:
            points (np.ndarray): An array of 2D points.

        Returns:
            np.ndarray: An array of 2D points representing the corners of the minimum area rectangle
                     in a specific order (clockwise or counterclockwise starting from the top-left corner).
        """
        bounding_box = cv2.minAreaRect(points)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2

        box = np.array(
            [points[index_a], points[index_b], points[index_c], points[index_d]]
        ).astype(np.int32)

        return box

    def _to_img(self) -> Dict[str, Image.Image]:
        """
        Converts the internal data to a PIL Image with detection and recognition results.

        Returns:
            Dict[Image.Image]: A dictionary containing two images: 'doc_preprocessor_res' and 'ocr_res_img'.
        """
        boxes = self["rec_polys"]
        txts = self["rec_texts"]
        image = self["doc_preprocessor_res"]["output_img"]
        h, w = image.shape[0:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_left = Image.fromarray(image_rgb)
        img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
        random.seed(0)
        draw_left = ImageDraw.Draw(img_left)
        for idx, (box, txt) in enumerate(zip(boxes, txts)):
            try:
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                box = np.array(box)
                if len(box) > 4:
                    pts = [(x, y) for x, y in box.tolist()]
                    draw_left.polygon(pts, outline=color, width=8)
                    box = self.get_minarea_rect(box)
                    height = int(0.5 * (max(box[:, 1]) - min(box[:, 1])))
                    box[:2, 1] = np.mean(box[:, 1])
                    box[2:, 1] = np.mean(box[:, 1]) + min(20, height)
                draw_left.polygon(box, fill=color)
                img_right_text = draw_box_txt_fine(
                    (w, h), box, txt, PINGFANG_FONT_FILE_PATH
                )
                pts = np.array(box, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_right_text, [pts], True, color, 1)
                img_right = cv2.bitwise_and(img_right, img_right_text)
            except:
                continue

        img_left = Image.blend(Image.fromarray(image_rgb), img_left, 0.5)
        img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))

        model_settings = self["model_settings"]
        res_img_dict = {f"ocr_res_img": img_show}
        if model_settings["use_doc_preprocessor"]:
            res_img_dict.update(**self["doc_preprocessor_res"].img)
        return res_img_dict

    def _to_str(self, *args, **kwargs) -> Dict[str, str]:
        """Converts the instance's attributes to a dictionary and then to a string.

        Args:
            *args: Additional positional arguments passed to the base class method.
            **kwargs: Additional keyword arguments passed to the base class method.

        Returns:
            Dict[str, str]: A dictionary with the instance's attributes converted to strings.
        """
        data = {}
        data["input_path"] = self["input_path"]
        data["model_settings"] = self["model_settings"]
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].str["res"]
        data["dt_polys"] = self["dt_polys"]
        data["text_det_params"] = self["text_det_params"]
        data["text_type"] = self["text_type"]
        data["textline_orientation_angles"] = self["textline_orientation_angles"]
        data["text_rec_score_thresh"] = self["text_rec_score_thresh"]
        data["rec_texts"] = self["rec_texts"]
        data["rec_scores"] = self["rec_scores"]
        data["rec_polys"] = self["rec_polys"]
        data["rec_boxes"] = self["rec_boxes"]
        
        return StrMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs) -> Dict[str, str]:
        """
        Converts the object's data to a JSON dictionary.

        Args:
            *args: Positional arguments passed to the JsonMixin._to_json method.
            **kwargs: Keyword arguments passed to the JsonMixin._to_json method.

        Returns:
            Dict[str, str]: A dictionary containing the object's data in JSON format.
        """
        data = {}
        data["input_path"] = self["input_path"]
        data["model_settings"] = self["model_settings"]
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].json["res"]
        data["dt_polys"] = self["dt_polys"]
        data["text_det_params"] = self["text_det_params"]
        data["text_type"] = self["text_type"]
        data["textline_orientation_angles"] = self["textline_orientation_angles"]
        data["text_rec_score_thresh"] = self["text_rec_score_thresh"]
        data["rec_texts"] = self["rec_texts"]
        data["rec_scores"] = self["rec_scores"]
        data["rec_polys"] = self["rec_polys"]
        data["rec_boxes"] = self["rec_boxes"]
        return JsonMixin._to_json(data, *args, **kwargs)


# Adds a function comment according to Google Style Guide
def draw_box_txt_fine(
    img_size: tuple, box: np.ndarray, txt: str, font_path: str
) -> np.ndarray:
    """
    Draws text in a box on an image with fine control over size and orientation.

    Args:
        img_size (tuple): The size of the output image (width, height).
        box (np.ndarray): A 4x2 numpy array defining the corners of the box in (x, y) order.
        txt (str): The text to draw inside the box.
        font_path (str): The path to the font file to use for drawing the text.

    Returns:
        np.ndarray: An image with the text drawn in the specified box.
    """
    box_height = int(
        math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
    )
    box_width = int(
        math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
    )

    if box_height > 2 * box_width and box_height > 30:
        img_text = Image.new("RGB", (box_height, box_width), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_height, box_width), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.ROTATE_270)
    else:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw_text = ImageDraw.Draw(img_text)
        if txt:
            font = create_font(txt, (box_width, box_height), font_path)
            draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

    pts1 = np.float32(
        [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
    )
    pts2 = np.array(box, dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_text = np.array(img_text, dtype=np.uint8)
    img_right_text = cv2.warpPerspective(
        img_text,
        M,
        img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return img_right_text
