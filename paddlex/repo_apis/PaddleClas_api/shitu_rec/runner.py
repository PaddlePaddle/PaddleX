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

from ..cls import ClsRunner


class ShiTuRecRunner(ClsRunner):
    """ShiTuRec Runner"""


def _extract_eval_metrics(stdout: str) -> dict:
    """extract evaluation metrics from training log

    Args:
        stdout (str): the training log

    Returns:
        dict: the training metric
    """
    import re

    _DP = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
    patterns = [
        r"\[Eval\]\[Epoch 0\]\[Avg\].*top1: (_dp), top5: (_dp)".replace("_dp", _DP),
        r"\[Eval\]\[Epoch 0\]\[Avg\].*recall1: (_dp), recall5: (_dp), mAP: (_dp)".replace(
            "_dp", _DP
        ),
    ]
    keys = [["val.top1", "val.top5"], ["recall1", "recall5", "mAP"]]

    metric_dict = dict()
    for pattern, key in zip(patterns, keys):
        pattern = re.compile(pattern)
        for line in stdout.splitlines():
            match = pattern.search(line)
            if match:
                for k, v in zip(key, map(float, match.groups())):
                    metric_dict[k] = v
    return metric_dict
