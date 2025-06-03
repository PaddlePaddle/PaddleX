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


import argparse
import os
import sys

from ....utils.deps import function_requires_deps, is_dep_available

if is_dep_available("pycocotools"):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_json_path", type=str, default="./bbox.json")
    parser.add_argument("--gt_json_path", type=str, default="./instance_val.json")

    args = parser.parse_args()
    return args


def json_eval_results(args):
    """
    cocoapi eval with already exists bbox.json
    """
    prediction_json_path = args.prediction_json_path
    gt_json_path = args.gt_json_path
    assert os.path.exists(
        prediction_json_path
    ), "The json directory:{} does not exist".format(prediction_json_path)
    cocoapi_eval(prediction_json_path, "bbox", anno_file=gt_json_path)


@function_requires_deps("pycocotools")
def cocoapi_eval(
    jsonfile,
    style,
    coco_gt=None,
    anno_file=None,
    max_dets=(100, 300, 1000),
    sigmas=None,
    use_area=True,
):
    """
    Args:
        jsonfile (str): Evaluation json file, eg: bbox.json
        style (str): COCOeval style, can be `bbox`
        coco_gt (str): Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file (str): COCO annotations file.
        max_dets (tuple): COCO evaluation maxDets.
        sigmas (nparray): keypoint labelling sigmas.
        use_area (bool): If gt annotations (eg. CrowdPose, AIC)
                         do not have 'area', please set use_area=False.
    """
    assert coco_gt is not None or anno_file is not None

    if coco_gt is None:
        coco_gt = COCO(anno_file)
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == "proposal":
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    elif style == "keypoints_crowd":
        coco_eval = COCOeval(coco_gt, coco_dt, style, sigmas, use_area)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # flush coco evaluation result
    sys.stdout.flush()
    return coco_eval.stats


if __name__ == "__main__":
    args = parse_args()
    json_eval_results(args)
