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

import collections
import math
import os
import time

import numpy as np
from tqdm import trange


def eval_segmentation(model, data_dir, batch_size=1):
    import cv2

    from .utils import Cityscapes, accuracy, calculate_area, f1_score, kappa, mean_iou

    assert os.path.isdir(data_dir), "The image_file_path:{} is not a directory.".format(
        data_dir
    )
    eval_dataset = Cityscapes(dataset_root=data_dir, mode="val")
    file_list = eval_dataset.file_list
    image_num = eval_dataset.num_samples
    num_classes = eval_dataset.num_classes
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0
    conf_mat_all = []
    twenty_percent_image_num = math.ceil(image_num * 0.2)
    start_time = 0
    end_time = 0
    average_inference_time = 0
    im_list = []
    label_list = []
    for image_label_path, i in zip(
        file_list, trange(image_num, desc="Inference Progress")
    ):
        if i == twenty_percent_image_num:
            start_time = time.time()
        im = cv2.imread(image_label_path[0])
        label = cv2.imread(image_label_path[1], cv2.IMREAD_GRAYSCALE)
        label_list.append(label)
        if batch_size == 1:
            result = model.predict(im)
            results = [result]
        else:
            im_list.append(im)
            # If the batch_size is not satisfied, the remaining pictures are formed into a batch
            if (i + 1) % batch_size != 0 and i != image_num - 1:
                continue
            results = model.batch_predict(im_list)
        if i == image_num - 1:
            end_time = time.time()
            average_inference_time = round(
                (end_time - start_time) / (image_num - twenty_percent_image_num), 4
            )
        for result, label in zip(results, label_list):
            pred = np.array(result.label_map).reshape(result.shape[0], result.shape[1])
            intersect_area, pred_area, label_area = calculate_area(
                pred, label, num_classes
            )
            intersect_area_all = intersect_area_all + intersect_area
            pred_area_all = pred_area_all + pred_area
            label_area_all = label_area_all + label_area
        im_list.clear()
        label_list.clear()

    class_iou, miou = mean_iou(intersect_area_all, pred_area_all, label_area_all)
    class_acc, oacc = accuracy(intersect_area_all, pred_area_all)
    kappa_res = kappa(intersect_area_all, pred_area_all, label_area_all)
    category_f1score = f1_score(intersect_area_all, pred_area_all, label_area_all)

    eval_metrics = collections.OrderedDict(
        zip(
            [
                "miou",
                "category_iou",
                "oacc",
                "category_acc",
                "kappa",
                "category_F1-score",
                "average_inference_time(s)",
            ],
            [
                miou,
                class_iou,
                oacc,
                class_acc,
                kappa_res,
                category_f1score,
                average_inference_time,
            ],
        )
    )
    return eval_metrics
