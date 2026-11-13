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

"""Per-model dataset wiring for the layout analysis module.

PP-DocLayoutV3 predicts instance masks, so it trains on ``COCOInstSegDataset``
and needs ``gt_poly``. PP-DocLayoutV4 predicts 4-point quads instead of masks,
so it trains on ``COCODetDataset`` and needs ``gt_quad``. Both need
``gt_read_order`` for the reading-order branch.
"""

DATASET_SPECS = {
    "PP-DocLayoutV3": {
        "dataset_type": "COCOInstSegDataset",
        "requires_segmentation": True,
        "data_fields": [
            "image",
            "gt_bbox",
            "gt_class",
            "gt_poly",
            "is_crowd",
            "gt_read_order",
        ],
    },
    "PP-DocLayoutV4": {
        "dataset_type": "COCODetDataset",
        # V4 needs no `segmentation`: it regresses quads, and `gt_quad` is
        # synthesized from the axis-aligned COCO `bbox` by COCODetDataset --
        # non-rectangular quads only ever come from RandomPerspectiveQuad.
        "requires_segmentation": False,
        "data_fields": [
            "image",
            "gt_bbox",
            "gt_quad",
            "gt_class",
            "is_crowd",
            "gt_read_order",
        ],
    },
}


def get_dataset_spec(model_name):
    """Return the ``(dataset_type, data_fields)`` pair for `model_name`.

    Args:
        model_name (str): the layout analysis model name, i.e. `Global.model`.

    Returns:
        tuple[str, list[str]]: the PaddleDetection dataset class name and the
            data fields the reader has to produce.

    Raises:
        ValueError: `model_name` has no registered spec.
    """
    if model_name not in DATASET_SPECS:
        raise ValueError(
            f"No layout analysis dataset spec registered for model {model_name!r}. "
            f"Known models: {sorted(DATASET_SPECS)}."
        )
    spec = DATASET_SPECS[model_name]
    return spec["dataset_type"], spec["data_fields"]


def requires_segmentation(model_name):
    """Whether `model_name` needs a `segmentation` field on every annotation.

    Kept separate from :func:`get_dataset_spec` so the existing two-tuple
    callers (trainer / evaluator / checker) stay untouched.

    Args:
        model_name (str): the layout analysis model name, i.e. `Global.model`.

    Returns:
        bool: True if `check_dataset` must reject annotations without masks.

    Raises:
        ValueError: `model_name` has no registered spec.
    """
    if model_name not in DATASET_SPECS:
        raise ValueError(
            f"No layout analysis dataset spec registered for model {model_name!r}. "
            f"Known models: {sorted(DATASET_SPECS)}."
        )
    return DATASET_SPECS[model_name]["requires_segmentation"]
