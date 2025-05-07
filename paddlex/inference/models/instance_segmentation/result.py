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

import numpy as np
from PIL import Image

from ....utils.deps import function_requires_deps, is_dep_available
from ...common.result import BaseCVResult, JsonMixin
from ...utils.color_map import get_colormap
from ..object_detection.result import draw_box

if is_dep_available("opencv-contrib-python"):
    import cv2


@function_requires_deps("opencv-contrib-python")
def draw_segm(im, masks, mask_info, alpha=0.7):
    """
    Draw segmentation on image
    """
    w_ratio = 0.4
    color_list = get_colormap(rgb=True)
    im = np.array(im).astype("float32")
    clsid2color = {}
    masks = np.array(masks)
    masks = masks.astype(np.uint8)
    for i in range(masks.shape[0]):
        mask, score, clsid = masks[i], mask_info[i]["score"], mask_info[i]["class_id"]

        if clsid not in clsid2color:
            color_index = i % len(color_list)
            clsid2color[clsid] = color_list[color_index]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        idx0 = np.minimum(idx[0], im.shape[0] - 1)
        idx1 = np.minimum(idx[1], im.shape[1] - 1)
        im[idx0, idx1, :] *= 1.0 - alpha
        im[idx0, idx1, :] += alpha * color_mask
        sum_x = np.sum(mask, axis=0)
        x = np.where(sum_x > 0.5)[0]
        sum_y = np.sum(mask, axis=1)
        y = np.where(sum_y > 0.5)[0]
        x0, x1, y0, y1 = x[0], x[-1], y[0], y[-1]
        cv2.rectangle(
            im, (x0, y0), (x1, y1), tuple(color_mask.astype("int32").tolist()), 1
        )
        bbox_text = "%s %.2f" % (mask_info[i]["label"], score)
        t_size = cv2.getTextSize(bbox_text, 0, 0.3, thickness=1)[0]
        cv2.rectangle(
            im,
            (x0, y0),
            (x0 + t_size[0], y0 - t_size[1] - 3),
            tuple(color_mask.astype("int32").tolist()),
            -1,
        )
        cv2.putText(
            im,
            bbox_text,
            (x0, y0 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    return Image.fromarray(im.astype("uint8"))


def restore_to_draw_masks(img_size, boxes, masks):
    """
    Restores extracted masks to the original shape and draws them on a blank image.

    """

    restored_masks = []

    for i, (box, mask) in enumerate(zip(boxes, masks)):
        restored_mask = np.zeros(img_size, dtype=np.uint8)
        x_min, y_min, x_max, y_max = map(lambda x: int(round(x)), box["coordinate"])
        restored_mask[y_min:y_max, x_min:x_max] = mask
        restored_masks.append(restored_mask)

    return np.array(restored_masks)


def draw_mask(im, boxes, np_masks, img_size):
    """
    Args:
        im (PIL.Image.Image): PIL image
        boxes (list): a list of dictionaries representing detection box information.
        np_masks (np.ndarray): shape:[N, im_h, im_w]
    Returns:
        im (PIL.Image.Image): visualized image
    """
    color_list = get_colormap(rgb=True)
    w_ratio = 0.4
    alpha = 0.7
    im = np.array(im).astype("float32")
    clsid2color = {}
    np_masks = restore_to_draw_masks(img_size, boxes, np_masks)
    im_h, im_w = im.shape[:2]
    np_masks = np_masks[:, :im_h, :im_w]
    for i in range(len(np_masks)):
        clsid, score = int(boxes[i]["cls_id"]), boxes[i]["score"]
        mask = np_masks[i]
        if clsid not in clsid2color:
            color_index = i % len(color_list)
            clsid2color[clsid] = color_list[color_index]
        color_mask = clsid2color[clsid]
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
        idx = np.nonzero(mask)
        color_mask = np.array(color_mask)
        im[idx[0], idx[1], :] *= 1.0 - alpha
        im[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(im.astype("uint8"))


class InstanceSegResult(BaseCVResult):
    """Save Result Transform"""

    def _to_img(self):
        """apply"""
        # image = self._img_reader.read(self["input_path"])
        image = Image.fromarray(self["input_img"])
        ori_img_size = list(image.size)[::-1]
        boxes = self["boxes"]
        masks = self["masks"]
        if next((True for item in self["boxes"] if "coordinate" in item), False):
            image = draw_mask(image, boxes, masks, ori_img_size)
            image = draw_box(image, boxes)
        else:
            image = draw_segm(image, masks, boxes)

        return {"res": image}

    def _to_str(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        data["masks"] = "..."
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs):
        data = copy.deepcopy(self)
        data.pop("input_img")
        return JsonMixin._to_json(data, *args, **kwargs)
