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


"""Process-local V4 COCO evaluation compatibility for PaddleDetection.

Executed as a script to avoid importing PaddleX in the training worker. The
external repository stays untouched. COCO box AP uses the enclosing rectangle;
this does not claim quadrilateral AP or reading-order accuracy.
"""

import functools
import importlib
import os
import runpy
import sys


def adapt_quad_results(convert):
    """Keep the existing converter semantics, accepting ordered quads as well."""

    @functools.wraps(convert)
    def wrapped(bboxes, *args, **kwargs):
        if bboxes.ndim == 2 and bboxes.shape[1] == 11:
            import numpy as np

            points = bboxes[:, 2:10].reshape(-1, 4, 2)
            bboxes = np.concatenate(
                [
                    bboxes[:, :2],
                    points.min(axis=1),
                    points.max(axis=1),
                    bboxes[:, 10:11],
                ],
                axis=1,
            )
        return convert(bboxes, *args, **kwargs)

    return wrapped


def main():
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path.insert(0, os.path.dirname(os.path.dirname(script)))
    json_results = importlib.import_module("ppdet.metrics.json_results")

    json_results.get_det_res_with_order = adapt_quad_results(
        json_results.get_det_res_with_order
    )
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
