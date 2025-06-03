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
import numpy as np
import os
import time
import collections


def topk_accuracy(topk_list, label_list):
    match_array = np.logical_or.reduce(topk_list == label_list, axis=1)
    topk_acc_score = match_array.sum() / match_array.shape[0]
    return topk_acc_score


def eval_classify(model, image_file_path, label_file_path, topk=5):
    from tqdm import trange
    import cv2
    import math

    result_list = []
    label_list = []
    image_label_dict = {}
    assert os.path.isdir(
        image_file_path
    ), "The image_file_path:{} is not a directory.".format(image_file_path)
    assert os.path.isfile(
        label_file_path
    ), "The label_file_path:{} is not a file.".format(label_file_path)
    assert isinstance(topk, int), "The tok:{} is not int type".format(topk)

    with open(label_file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            items = line.strip().split()
            image_name = items[0]
            label = items[1]
            image_label_dict[image_name] = int(label)
    images_num = len(image_label_dict)
    twenty_percent_images_num = math.ceil(images_num * 0.2)
    start_time = 0
    end_time = 0
    average_inference_time = 0
    scores = collections.OrderedDict()
    for (image, label), i in zip(
        image_label_dict.items(), trange(images_num, desc="Inference Progress")
    ):
        if i == twenty_percent_images_num:
            start_time = time.time()

        label_list.append([label])
        image_path = os.path.join(image_file_path, image)
        im = cv2.imread(image_path)
        result = model.predict(im, topk)
        result_list.append(result.label_ids)
        if i == images_num - 1:
            end_time = time.time()
    average_inference_time = round(
        (end_time - start_time) / (images_num - twenty_percent_images_num), 4
    )
    topk_acc_score = topk_accuracy(np.array(result_list), np.array(label_list))
    if topk == 1:
        scores.update({"topk1": topk_acc_score})
        scores.update({"topk1_average_inference_time(s)": average_inference_time})
    elif topk == 5:
        scores.update({"topk5": topk_acc_score})
        scores.update({"topk5_average_inference_time(s)": average_inference_time})
    return scores
