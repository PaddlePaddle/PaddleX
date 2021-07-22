# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score as accuracy_metric
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import binarize

import numpy as np

__all__ = ["multi_hot_encode", "hamming_distance", "accuracy_score", "precision_recall_fscore", "mean_average_precision"]


def multi_hot_encode(logits, threshold=0.5):
    """
    Encode logits to multi-hot by elementwise for multilabel
    """

    return binarize(logits, threshold)


def hamming_distance(output, target):
    """
    Soft metric based label for multilabel classification
    Returns:
        The smaller the return value is, the better model is.
    """

    return hamming_loss(target, output)


def accuracy_score(output, target, base="sample"):
    """
    Hard metric for multilabel classification
    Args:
        output:
        target:
        base: ["sample", "label"], default="sample"
            if "sample", return metric score based sample,
            if "label", return metric score based label.
    Returns:
        accuracy:
    """

    assert base in ["sample", "label"], 'must be one of ["sample", "label"]'

    if base == "sample":
        accuracy = accuracy_metric(target, output)
    elif base == "label":
        mcm = multilabel_confusion_matrix(target, output)
        tns = mcm[:, 0, 0]
        fns = mcm[:, 1, 0]
        tps = mcm[:, 1, 1]
        fps = mcm[:, 0, 1]

        accuracy = (sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fns) + sum(fps))

    return accuracy


def precision_recall_fscore(output, target):
    """
    Metric based label for multilabel classification
    Returns:
        precisions:
        recalls:
        fscores:
    """

    precisions, recalls, fscores, _ = precision_recall_fscore_support(target, output)

    return precisions, recalls, fscores


def mean_average_precision(logits, target):
    """
    Calculate average precision
    Args:
        logits: probability from network before sigmoid or softmax
        target: ground truth, 0 or 1
    """
    if not (isinstance(logits, np.ndarray) and isinstance(target, np.ndarray)):
        raise TypeError("logits and target should be np.ndarray.")

    aps = []
    for i in range(target.shape[1]):
        ap = average_precision_score(target[:, i], logits[:, i])
        aps.append(ap)

    return np.mean(aps)
