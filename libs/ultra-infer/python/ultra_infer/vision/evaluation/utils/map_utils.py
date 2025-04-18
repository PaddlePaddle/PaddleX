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
import os


def draw_pr_curve(
    precision,
    recall,
    iou=0.5,
    out_dir="pr_curve",
    file_name="precision_recall_curve.jpg",
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_path = os.path.join(out_dir, file_name)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        # logger.error('Matplotlib not found, plaese install matplotlib.'
        #              'for example: `pip install matplotlib`.')
        raise e
    plt.cla()
    plt.figure("P-R Curve")
    plt.title("Precision/Recall Curve(IoU={})".format(iou))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.plot(recall, precision)
    plt.savefig(output_path)
