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
import sys
import cv2
import math
import random
import tempfile
import subprocess
import numpy as np
from PIL import Image, ImageDraw

from .base import CVResult
from ...utils import logging
from .ocr import draw_box_txt_fine
from ...utils.fonts import PINGFANG_FONT_FILE_PATH


class FormulaRecResult(CVResult):
    def _to_str(self, *args, **kwargs):
        return super()._to_str(*args, **kwargs).replace("\\\\", "\\")

    def _to_img(
        self,
    ):
        """Draw formula on image"""
        try:
            import fitz
        except:
            logging.warning(
                "Please refer to 2.3 Formula Recognition Pipeline Visualization in Formula Recognition Pipeline Tutorial to install the LaTeX rendering engine at first."
            )
            return None
        image = self._img_reader.read(self["input_path"])
        rec_formula = str(self["rec_text"])
        image = np.array(image.convert("RGB"))
        xywh = crop_white_area(image)
        if xywh is not None:
            x, y, w, h = xywh
            image = image[y : y + h, x : x + w]
        image = Image.fromarray(image)
        image_width, image_height = image.size
        box = [[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]]
        try:
            img_formula = draw_formula_module(
                image.size, box, rec_formula, is_debug=False
            )
            img_formula = Image.fromarray(img_formula)
            render_width, render_height = img_formula.size
            resize_height = render_height
            resize_width = int(resize_height * image_width / image_height)
            image = image.resize((resize_width, resize_height), Image.ANTIALIAS)

            new_image_width = image.width + int(render_width) + 10
            new_image = Image.new(
                "RGB", (new_image_width, render_height), (255, 255, 255)
            )
            new_image.paste(image, (0, 0))
            new_image.paste(img_formula, (image.width + 10, 0))
            return new_image
        except subprocess.CalledProcessError as e:
            logging.warning(
                "Please refer to 2.3 Formula Recognition Pipeline Visualization in Formula Recognition Pipeline Tutorial to install the LaTeX rendering engine at first."
            )
            return None


class FormulaResult(CVResult):
    _HARD_FLAG = False

    def _to_str(self, *args, **kwargs):
        return super()._to_str(*args, **kwargs).replace("\\\\", "\\")

    def _to_img(
        self,
    ):
        """draw formula result"""
        try:
            import fitz
        except:
            logging.warning(
                "Please refer to 2.3 Formula Recognition Pipeline Visualization in Formula Recognition Pipeline Tutorial to install the LaTeX rendering engine at first."
            )
            return None

        boxes = self["dt_polys"]
        formulas = self["rec_formula"]
        image = self._img_reader.read(self["input_path"])
        if self._HARD_FLAG:
            image_np = np.array(image)
            image = Image.fromarray(image_np[:, :, ::-1])
        h, w = image.height, image.width
        img_left = image.copy()
        img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
        random.seed(0)
        draw_left = ImageDraw.Draw(img_left)

        if formulas is None or len(formulas) != len(boxes):
            formulas = [None] * len(boxes)
        for idx, (box, formula) in enumerate(zip(boxes, formulas)):
            try:
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                box = np.array(box)
                pts = [(x, y) for x, y in box.tolist()]
                draw_left.polygon(pts, outline=color, width=8)
                draw_left.polygon(box, fill=color)
                img_right_text = draw_box_formula_fine(
                    (w, h),
                    box,
                    formula,
                    is_debug=False,
                )
                pts = np.array(box, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_right_text, [pts], True, color, 1)
                img_right = cv2.bitwise_and(img_right, img_right_text)
            except subprocess.CalledProcessError as e:
                continue

        img_left = Image.blend(image, img_left, 0.5)
        img_show = Image.new("RGB", (int(w * 2), h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
        return img_show


def get_align_equation(equation):
    is_align = False
    equation = str(equation) + "\n"
    begin_dict = [
        r"begin{align}",
        r"begin{align*}",
    ]
    for begin_sym in begin_dict:
        if begin_sym in equation:
            is_align = True
            break
    if not is_align:
        equation = (
            r"\begin{equation}"
            + "\n"
            + equation.strip()
            + r"\nonumber"
            + "\n"
            + r"\end{equation}"
            + "\n"
        )
    return equation


def generate_tex_file(tex_file_path, equation):
    with open(tex_file_path, "w") as fp:
        start_template = (
            r"\documentclass{article}" + "\n"
            r"\usepackage{cite}" + "\n"
            r"\usepackage{amsmath,amssymb,amsfonts}" + "\n"
            r"\usepackage{graphicx}" + "\n"
            r"\usepackage{textcomp}" + "\n"
            r"\DeclareMathSizes{14}{14}{9.8}{7}" + "\n"
            r"\pagestyle{empty}" + "\n"
            r"\begin{document}" + "\n"
            r"\begin{large}" + "\n"
        )
        fp.write(start_template)
        equation = get_align_equation(equation)
        fp.write(equation)
        end_template = r"\end{large}" + "\n" r"\end{document}" + "\n"
        fp.write(end_template)


def generate_pdf_file(tex_path, pdf_dir, is_debug=False):
    if os.path.exists(tex_path):
        if is_debug:
            subprocess.check_call(
                [
                    "pdflatex",
                    "-halt-on-error",
                    "-output-directory={}".format(pdf_dir),
                    tex_path,
                ]
            )
        else:
            devNull = open(os.devnull, "w")
            subprocess.check_call(
                [
                    "pdflatex",
                    "-halt-on-error",
                    "-output-directory={}".format(pdf_dir),
                    tex_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )


def crop_white_area(image):
    image = np.array(image).astype("uint8")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        return [x, y, w, h]
    else:
        return None


def pdf2img(pdf_path, img_path, is_padding=False):
    import fitz

    pdfDoc = fitz.open(pdf_path)
    if pdfDoc.page_count != 1:
        return None
    for pg in range(pdfDoc.page_count):
        page = pdfDoc[pg]
        rotate = int(0)
        zoom_x = 2
        zoom_y = 2
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        pix._writeIMG(img_path, 7, 100)
        img = cv2.imread(img_path)
        xywh = crop_white_area(img)

        if xywh is not None:
            x, y, w, h = xywh
            img = img[y : y + h, x : x + w]
            if is_padding:
                img = cv2.copyMakeBorder(
                    img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
            return img
    return None


def draw_formula_module(img_size, box, formula, is_debug=False):
    """draw box formula for module"""
    box_width, box_height = img_size
    with tempfile.TemporaryDirectory() as td:
        tex_file_path = os.path.join(td, "temp.tex")
        pdf_file_path = os.path.join(td, "temp.pdf")
        img_file_path = os.path.join(td, "temp.jpg")
        generate_tex_file(tex_file_path, formula)
        if os.path.exists(tex_file_path):
            generate_pdf_file(tex_file_path, td, is_debug)
        formula_img = None
        if os.path.exists(pdf_file_path):
            formula_img = pdf2img(pdf_file_path, img_file_path, is_padding=False)
        if formula_img is not None:
            return formula_img
        else:
            img_right_text = draw_box_txt_fine(
                img_size, box, "渲染失败", PINGFANG_FONT_FILE_PATH
            )
        return img_right_text


def draw_box_formula_fine(img_size, box, formula, is_debug=False):
    """draw box formula for pipeline"""
    box_height = int(
        math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
    )
    box_width = int(
        math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
    )
    with tempfile.TemporaryDirectory() as td:
        tex_file_path = os.path.join(td, "temp.tex")
        pdf_file_path = os.path.join(td, "temp.pdf")
        img_file_path = os.path.join(td, "temp.jpg")
        generate_tex_file(tex_file_path, formula)
        if os.path.exists(tex_file_path):
            generate_pdf_file(tex_file_path, td, is_debug)
        formula_img = None
        if os.path.exists(pdf_file_path):
            formula_img = pdf2img(pdf_file_path, img_file_path, is_padding=False)
        if formula_img is not None:
            formula_h, formula_w = formula_img.shape[:-1]
            resize_height = box_height
            resize_width = formula_w * resize_height / formula_h
            formula_img = cv2.resize(
                formula_img, (int(resize_width), int(resize_height))
            )
            formula_h, formula_w = formula_img.shape[:-1]
            pts1 = np.float32(
                [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
            )
            pts2 = np.array(box, dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts1, pts2)
            formula_img = np.array(formula_img, dtype=np.uint8)
            img_right_text = cv2.warpPerspective(
                formula_img,
                M,
                img_size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
        else:
            img_right_text = draw_box_txt_fine(
                img_size, box, "渲染失败", PINGFANG_FONT_FILE_PATH
            )
        return img_right_text
