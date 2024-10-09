import os
from pathlib import Path


def get_ocr_res(pipeline, input):
    """get ocr res"""
    ocr_res_list = []
    if isinstance(input, list):
        img = [im["img"] for im in input]
    elif isinstance(input, dict):
        img = input["img"]
    else:
        img = input
    for ocr_res in pipeline(img):
        ocr_res_list.append(ocr_res)
    return ocr_res_list