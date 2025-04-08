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
import json
import os

import paddle

from ....utils import logging


def parse_args():
    """Parse all arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_json_path", type=str, default="./pre_res.json")
    parser.add_argument("--gt_val_path", type=str, default="./val.txt")
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--num_classes", type=int)

    args = parser.parse_args()
    return args


class AvgMetrics(paddle.nn.Layer):
    """Average metrics"""

    def __init__(self):
        super().__init__()
        self.avg_meters = {}

    @property
    def avg(self):
        """Return average value of each metric"""
        if self.avg_meters:
            for metric_key in self.avg_meters:
                return self.avg_meters[metric_key].avg

    @property
    def avg_info(self):
        """Return a formatted string of average values and names"""
        return ", ".join([self.avg_meters[key].avg_info for key in self.avg_meters])


class TopkAcc(AvgMetrics):
    """Top-k accuracy metric"""

    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk
        self.warned = False

    def forward(self, x, label):
        """forward function"""
        if isinstance(x, dict):
            x = x["logits"]

        output_dims = x.shape[-1]

        metric_dict = dict()
        for idx, k in enumerate(self.topk):
            if output_dims < k:
                if not self.warned:
                    msg = f"The output dims({output_dims}) is less than k({k}), so the Top-{k} metric is meaningless."
                    logging.info(msg)
                    self.warned = True
                metric_dict[f"top{k}"] = 1
            else:
                metric_dict[f"top{k}"] = paddle.metric.accuracy(x, label, k=k).item()
        return metric_dict


def prase_pt_info(pt_info, num_classes):
    """Parse prediction information to probability vector"""
    pre_list = [0.0] * num_classes
    for idx, val in zip(pt_info["class_ids"], pt_info["scores"]):
        pre_list[idx] = val
    return pre_list


def main(args):
    """main function"""
    with open(args.prediction_json_path, "r") as fp:
        predication_result = json.load(fp)
    gt_info = {}

    pred = []
    label = []
    for line in open(args.gt_val_path):
        img_file, gt_label = line.strip().split(" ")
        img_file = img_file.split("/")[-1]
        gt_info[img_file] = int(gt_label)
    for pt_info in predication_result:
        img_file = os.path.relpath(pt_info["file_name"], args.image_dir)
        pred.append(prase_pt_info(pt_info, args.num_classes))
        label.append([gt_info[img_file]])
    metric_dict = TopkAcc()(paddle.to_tensor(pred), paddle.to_tensor(label))
    logging.info(metric_dict)


if __name__ == "__main__":
    args = parse_args()
    main(args)
