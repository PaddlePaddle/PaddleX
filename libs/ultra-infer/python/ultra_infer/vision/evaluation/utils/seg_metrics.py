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


def f1_score(intersect_area, pred_area, label_area):
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


def calculate_area(pred, label, num_classes, ignore_index=255):
    """
    Calculate intersect, prediction and label area

    Args:
        pred (np.ndarray): The prediction by model.
        label (np.ndarray): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.

    Returns:
        Numpy Array: The intersection area of prediction and the ground on all class.
        Numpy Array: The prediction area on all class.
        Numpy Array: The ground truth area on all class
    """
    if not pred.shape == label.shape:
        raise ValueError(
            "Shape of `pred` and `label should be equal, "
            "but there are {} and {}.".format(pred.shape, label.shape)
        )

    mask = label != ignore_index
    pred = pred + 1
    label = label + 1
    pred = pred * mask
    label = label * mask
    pred = np.eye(num_classes + 1)[pred]
    label = np.eye(num_classes + 1)[label]
    pred = pred[:, 1:]
    label = label[:, 1:]

    pred_area = []
    label_area = []
    intersect_area = []

    for i in range(num_classes):
        pred_i = pred[:, :, i]
        label_i = label[:, :, i]
        pred_area_i = np.sum(pred_i)
        label_area_i = np.sum(label_i)
        intersect_area_i = np.sum(pred_i * label_i)
        pred_area.append(pred_area_i)
        label_area.append(label_area_i)
        intersect_area.append(intersect_area_i)
    return np.array(intersect_area), np.array(pred_area), np.array(label_area)


def mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.

    Args:
        intersect_area (np.ndarray): The intersection area of prediction and ground truth on all classes.
        pred_area (np.ndarray): The prediction area on all classes.
        label_area (np.ndarray): The ground truth area on all classes.

    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou


def accuracy(intersect_area, pred_area):
    """
    Calculate accuracy

    Args:
        intersect_area (np.ndarray): The intersection area of prediction and ground truth on all classes..
        pred_area (np.ndarray): The prediction area on all classes.

    Returns:
        np.ndarray: accuracy on all classes.
        float: mean accuracy.
    """
    class_acc = []
    for i in range(len(intersect_area)):
        if pred_area[i] == 0:
            acc = 0
        else:
            acc = intersect_area[i] / pred_area[i]
        class_acc.append(acc)
    macc = np.sum(intersect_area) / np.sum(pred_area)
    return np.array(class_acc), macc


def kappa(intersect_area, pred_area, label_area):
    """
    Calculate kappa coefficient

    Args:
        intersect_area (np.ndarray): The intersection area of prediction and ground truth on all classes..
        pred_area (np.ndarray): The prediction area on all classes.
        label_area (np.ndarray): The ground truth area on all classes.

    Returns:
        float: kappa coefficient.
    """
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    kappa = (po - pe) / (1 - pe)
    return kappa
