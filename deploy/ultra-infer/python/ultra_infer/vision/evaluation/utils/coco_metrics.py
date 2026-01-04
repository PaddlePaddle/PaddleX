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
from __future__ import division
from __future__ import print_function

import copy
import sys
from collections import OrderedDict
from .coco_utils import get_infer_results, cocoapi_eval


class COCOMetric(object):
    def __init__(self, coco_gt, **kwargs):
        self.clsid2catid = {
            i: cat["id"] for i, cat in enumerate(coco_gt.loadCats(coco_gt.getCatIds()))
        }
        self.coco_gt = coco_gt
        self.classwise = kwargs.get("classwise", False)
        self.bias = 0
        self.reset()

    def reset(self):
        # only bbox and mask evaluation support currently
        self.details = {
            "gt": copy.deepcopy(self.coco_gt.dataset),
            "bbox": [],
            "mask": [],
        }
        self.eval_stats = {}

    def update(self, im_id, outputs):
        outs = {}
        # outputs Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v

        outs["im_id"] = im_id
        infer_results = get_infer_results(outs, self.clsid2catid, bias=self.bias)
        self.details["bbox"] += infer_results["bbox"] if "bbox" in infer_results else []
        self.details["mask"] += infer_results["mask"] if "mask" in infer_results else []

    def accumulate(self):
        if len(self.details["bbox"]) > 0:
            bbox_stats = cocoapi_eval(
                copy.deepcopy(self.details["bbox"]),
                "bbox",
                coco_gt=self.coco_gt,
                classwise=self.classwise,
            )
            self.eval_stats["bbox"] = bbox_stats
            sys.stdout.flush()

        if len(self.details["mask"]) > 0:
            seg_stats = cocoapi_eval(
                copy.deepcopy(self.details["mask"]),
                "segm",
                coco_gt=self.coco_gt,
                classwise=self.classwise,
            )
            self.eval_stats["mask"] = seg_stats
            sys.stdout.flush()

    def log(self):
        pass

    def get(self):
        if "bbox" not in self.eval_stats:
            return {"bbox_mmap": 0.0}
        if "mask" in self.eval_stats:
            return OrderedDict(
                zip(
                    ["bbox_mmap", "segm_mmap"],
                    [self.eval_stats["bbox"][0], self.eval_stats["mask"][0]],
                )
            )
        else:
            return {"bbox_mmap": self.eval_stats["bbox"][0]}
