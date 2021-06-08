# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle


def loss_computation(logits_list, labels, losses):
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_i = losses['types'][i]
        loss_list.append(losses['coef'][i] * loss_i(logits, labels))

    return loss_list


def f1_score(intersect_area, pred_area, label_area):
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    class_f1_sco = []
    for i in range(len(intersect_area)):
        if pred_area[i] + label_area[i] == 0:
            f1_sco = 0
        elif pred_area[i] == 0:
            f1_sco = 0
        else:
            prec = intersect_area[i] / pred_area[i]
            rec = intersect_area[i] / label_area[i]
            f1_sco = 2 * prec * rec / (prec + rec)
        class_f1_sco.append(f1_sco)
    return np.array(class_f1_sco)


def confusion_matrix(pred, label, num_classes, ignore_index=255):
    label = paddle.transpose(label, (0, 2, 3, 1))
    pred = paddle.transpose(pred, (0, 2, 3, 1))
    mask = label != ignore_index

    label = paddle.masked_select(label, mask)
    pred = paddle.masked_select(pred, mask)

    cat_matrix = num_classes * label + pred
    conf_mat = paddle.histogram(
        cat_matrix,
        bins=num_classes * num_classes,
        min=0,
        max=num_classes * num_classes - 1).reshape([num_classes, num_classes])

    return conf_mat
