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

import copy
import math
import os
import re
import subprocess
import tempfile
from typing import List, Optional

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

from ....utils import logging
from ....utils.deps import function_requires_deps, is_dep_available
from ....utils.file_interface import custom_open
from ....utils.fonts import PINGFANG_FONT_FILE_PATH
from ...common.result import BaseCVResult, JsonMixin

if is_dep_available("opencv-contrib-python"):
    import cv2
if is_dep_available("pypdfium2"):
    import pypdfium2 as pdfium


class FormulaRecResult(BaseCVResult):

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        _str = JsonMixin._to_str(data, *args, **kwargs)["res"]
        return {"res": _str}

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)

    def _to_img(
        self,
    ) -> Image.Image:
        """
        Draws a recognized formula on an image.

        This method processes an input image to recognize and render a LaTeX formula.
        It overlays the rendered formula onto the input image and returns the combined image.
        If the LaTeX rendering engine is not installed or a syntax error is detected,
        it logs a warning and returns the original image.

        Returns:
            Image.Image: An image with the recognized formula rendered alongside the original image.
        """
        image = Image.fromarray(self["input_img"])
        try:
            env_valid()
        except subprocess.CalledProcessError as e:
            logging.warning(
                "Please refer to 2.3 Formula Recognition Pipeline Visualization in Formula Recognition Pipeline Tutorial to install the LaTeX rendering engine at first."
            )
            return {"res": image}

        rec_formula = str(self["rec_formula"])
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
            image = image.resize((resize_width, resize_height), Image.LANCZOS)

            new_image_width = image.width + int(render_width) + 10
            new_image = Image.new(
                "RGB", (new_image_width, render_height), (255, 255, 255)
            )
            new_image.paste(image, (0, 0))
            new_image.paste(img_formula, (image.width + 10, 0))
            return {"res": new_image}
        except subprocess.CalledProcessError as e:
            logging.warning("Syntax error detected in formula, rendering failed.")
            return {"res": image}


def get_align_equation(equation: str) -> str:
    """
    Wraps an equation in LaTeX environment tags if not already aligned.

    This function checks if a given LaTeX equation contains any alignment tags (`align` or `align*`).
    If the equation does not contain these tags, it wraps the equation in `equation` and `nonumber` tags.

    Args:
        equation (str): The LaTeX equation to be checked and potentially modified.

    Returns:
        str: The modified equation with appropriate LaTeX tags for alignment.
    """
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


def add_text_for_zh_formula(formula: str) -> str:
    pattern = re.compile(r"([^\x00-\x7F]+)")

    def replacer(match):
        return f"\\text{{{match.group(1)}}}"

    replaced_formula = pattern.sub(replacer, formula)

    return replaced_formula


def generate_tex_file(tex_file_path: str, equation: str) -> None:
    """
    Generates a LaTeX file containing a specific equation.

    This function creates a LaTeX file at the specified file path, writing the necessary
    LaTeX preamble and wrapping the provided equation in a document structure. The equation
    is processed to ensure it includes alignment tags if necessary.

    Args:
        tex_file_path (str): The file path where the LaTeX file will be saved.
        equation (str): The LaTeX equation to be written into the file.
    """
    with custom_open(tex_file_path, "w") as fp:
        start_template = (
            r"\documentclass[varwidth]{standalone}" + "\n"
            r"\usepackage{cite}" + "\n"
            r"\usepackage{amsmath,amssymb,amsfonts,upgreek}" + "\n"
            r"\usepackage{graphicx}" + "\n"
            r"\usepackage{textcomp}" + "\n"
            r"\usepackage{xeCJK}" + "\n"
            r"\DeclareMathSizes{14}{14}{9.8}{7}" + "\n"
            r"\pagestyle{empty}" + "\n"
            r"\begin{document}" + "\n"
            r"\begin{large}" + "\n"
        )
        fp.write(start_template)
        equation = add_text_for_zh_formula(equation)
        equation = get_align_equation(equation)
        fp.write(equation)
        end_template = r"\end{large}" + "\n" r"\end{document}" + "\n"
        fp.write(end_template)


def generate_pdf_file(
    tex_path: str, pdf_dir: str, is_debug: bool = False
) -> Optional[bool]:
    """
    Generates a PDF file from a LaTeX file using pdflatex.

    This function checks if the specified LaTeX file exists, and then runs pdflatex to generate a PDF file
    in the specified directory. It can run in debug mode to show detailed output or in silent mode.

    Args:
        tex_path (str): The path to the LaTeX file.
        pdf_dir (str): The directory where the PDF file will be saved.
        is_debug (bool, optional): If True, runs pdflatex with detailed output. Defaults to False.

    Returns:
        Optional[bool]: Returns True if the PDF was generated successfully, False if the LaTeX file does not exist,
                        and None if an error occurred during the pdflatex execution.
    """
    if os.path.exists(tex_path):
        command = "xelatex -interaction=nonstopmode -halt-on-error -output-directory={} {}".format(
            pdf_dir, tex_path
        )
        if is_debug:
            subprocess.check_call(command, shell=True)
        else:
            custom_open(os.devnull, "w")
            subprocess.check_call(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
            )


@function_requires_deps("opencv-contrib-python")
def crop_white_area(image: np.ndarray) -> Optional[List[int]]:
    """
    Finds and returns the bounding box of the non-white area in an image.

    This function converts an image to grayscale and uses binary thresholding to
    find contours. It then calculates the bounding rectangle around the non-white
    areas of the image.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        Optional[List[int]]: A list [x, y, w, h] representing the bounding box of
                             the non-white area, or None if no such area is found.
    """
    image = np.array(image).astype("uint8")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        return [x, y, w, h]
    else:
        return None


@function_requires_deps("pypdfium2", "opencv-contrib-python")
def pdf2img(pdf_path: str, img_path: str, is_padding: bool = False):
    """
    Converts a single-page PDF to an image, optionally cropping white areas and adding padding.

    Args:
        pdf_path (str): The path to the PDF file.
        img_path (str): The path where the image will be saved.
        is_padding (bool): If True, adds a 30-pixel white padding around the image.

    Returns:
        np.ndarray: The resulting image as a NumPy array, or None if the PDF is not single-page.
    """
    pdfDoc = pdfium.PdfDocument(pdf_path)
    if len(pdfDoc) != 1:
        return None
    for page in pdfDoc:
        rotate = int(0)
        zoom = 2
        img = page.render(scale=zoom, rotation=rotate).to_pil()
        img = img.convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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


def draw_formula_module(
    img_size: tuple, box: list, formula: str, is_debug: bool = False
):
    """
    Draw box formula for module.

    Args:
        img_size (tuple): The size of the image as (width, height).
        box (list): The coordinates for the bounding box.
        formula (str): The LaTeX formula to render.
        is_debug (bool): If True, retains intermediate files for debugging purposes.

    Returns:
        np.ndarray: The resulting image with the formula or an error message.
    """
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
                img_size, box, "Rendering Failed", PINGFANG_FONT_FILE_PATH
            )
        return img_right_text


def env_valid() -> bool:
    """
    Validates if the environment is correctly set up to convert LaTeX formulas to images.

    Returns:
        bool: True if the environment is valid and the conversion is successful, False otherwise.
    """
    with tempfile.TemporaryDirectory() as td:
        tex_file_path = os.path.join(td, "temp.tex")
        pdf_file_path = os.path.join(td, "temp.pdf")
        img_file_path = os.path.join(td, "temp.jpg")
        formula = "a+b=c"
        is_debug = False
        generate_tex_file(tex_file_path, formula)
        if os.path.exists(tex_file_path):
            generate_pdf_file(tex_file_path, td, is_debug)
        if os.path.exists(pdf_file_path):
            formula_img = pdf2img(pdf_file_path, img_file_path, is_padding=False)


@function_requires_deps("opencv-contrib-python")
def draw_box_txt_fine(img_size: tuple, box: list, txt: str, font_path: str):
    """
    Draw box text.

    Args:
        img_size (tuple): Size of the image as (width, height).
        box (list): List of four points defining the box, each point is a tuple (x, y).
        txt (str): The text to draw inside the box.
        font_path (str): Path to the font file to be used for drawing text.

    Returns:
        np.ndarray: Image array with the text drawn and transformed to fit the box.
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


def create_font(txt: str, sz: tuple, font_path: str) -> ImageFont.FreeTypeFont:
    """
    Creates a font object with a size that ensures the text fits within the specified dimensions.

    Args:
        txt (str): The text to fit.
        sz (tuple): The target size as (width, height).
        font_path (str): The path to the font file.

    Returns:
        ImageFont.FreeTypeFont: A PIL font object at the appropriate size.
    """
    font_size = int(sz[1] * 0.8)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    if int(PIL.__version__.split(".")[0]) < 10:
        length = font.getsize(txt)[0]
    else:
        length = font.getlength(txt)

    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font
