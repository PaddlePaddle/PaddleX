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
import random

import numpy as np
from PIL import Image

from .....utils.deps import function_requires_deps, is_dep_available
from ....common.result import BaseCVResult, JsonMixin
from ....utils.color_map import get_colormap

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
        mask = masks[i]
        clsid = random.randint(0, len(get_colormap(rgb=True)) - 1)

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
        # draw box prompt
        if mask_info[i]["label"] == "box_prompt":
            x0, y0, x1, y1 = mask_info[i]["prompt"]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            cv2.rectangle(
                im, (x0, y0), (x1, y1), tuple(color_mask.astype("int32").tolist()), 1
            )
            bbox_text = "%s" % mask_info[i]["label"]
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
        elif mask_info[i]["label"] == "point_prompt":
            x, y = mask_info[i]["prompt"]
            bbox_text = "%s" % mask_info[i]["label"]
            t_size = cv2.getTextSize(bbox_text, 0, 0.3, thickness=1)[0]
            cv2.circle(
                im,
                (x, y),
                1,
                (255, 255, 255),
                4,
            )
            cv2.putText(
                im,
                bbox_text,
                (x - t_size[0] // 2, y - t_size[1] - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
        else:
            raise NotImplementedError(
                f"Prompt type {mask_info[i]['label']} not implemented."
            )
    return Image.fromarray(im.astype("uint8"))


class SAMSegResult(BaseCVResult):
    """Save Result Transform for SAM"""

    def __init__(self, data: dict) -> None:

        data["masks"] = [mask.squeeze(0) for mask in list(data["masks"])]

        prompts = data["prompts"]
        assert isinstance(prompts, dict) and len(prompts) == 1
        prompt_type, prompts = list(prompts.items())[0]
        mask_infos = [
            {
                "label": prompt_type,
                "prompt": p,
            }
            for p in prompts
        ]
        data["mask_infos"] = mask_infos
        assert len(data["masks"]) == len(mask_infos)

        super().__init__(data)

    def _to_img(self):
        """apply"""
        image = Image.fromarray(self["input_img"])
        mask_infos = self["mask_infos"]
        masks = self["masks"]
        image = draw_segm(image, masks, mask_infos)
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
