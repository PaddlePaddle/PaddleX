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

"""Shared utilities for DocVLM predictors."""

from typing import Any, Dict, List

from .constants import MODEL_GROUP


def is_in_group(model_name: str, group_name: str) -> bool:
    """Check if model_name belongs to the given model group."""
    return model_name in MODEL_GROUP.get(group_name, set())


def format_doc_vlm_result_dict(
    model_preds: Any, src_data: List[dict], add_input_path: bool = True
) -> Dict[str, List]:
    """Format model predictions and source data into a result dict.

    Args:
        model_preds: Model predictions (single or list).
        src_data: Source data list.
        add_input_path: If True, add input_path from image when image is a path string.

    Returns:
        Dict with keys from src_data plus "result" containing predictions.
    """
    if not isinstance(model_preds, list):
        model_preds = [model_preds]
    if not isinstance(src_data, list):
        src_data = [src_data]

    input_info = []
    for data in src_data:
        data = dict(data)
        if add_input_path:
            image = data.get("image", None)
            if isinstance(image, str):
                data["input_path"] = image
        input_info.append(data)

    if len(model_preds) != len(input_info):
        raise ValueError(
            f"Model predicts {len(model_preds)} results while src data has {len(input_info)} samples."
        )

    rst_format_dict = {k: [] for k in input_info[0].keys()}
    rst_format_dict["result"] = []

    for data_sample, model_pred in zip(input_info, model_preds):
        for k in data_sample.keys():
            rst_format_dict[k].append(data_sample[k])
        rst_format_dict["result"].append(model_pred)

    return rst_format_dict
