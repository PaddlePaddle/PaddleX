# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import pytest

from paddlex.inference.pipelines.layout_parsing.pipeline_v2 import (
    _LayoutParsingPipelineV2,
)


def _parse_single_table(image, bbox):
    pipeline = object.__new__(_LayoutParsingPipelineV2)
    page = pipeline.get_layout_parsing_objects(
        image=image,
        region_block_ocr_idx_map={"region_to_block_map": {0: [0]}},
        region_det_res={"boxes": [{"coordinate": [0, 0, 1, 1]}]},
        overall_ocr_res={},
        layout_det_res={"boxes": [{"label": "table", "coordinate": bbox}]},
        table_res_list=[{"pred_html": "<table></table>"}],
        seal_res_list=[],
        chart_res_list=[],
        text_rec_model=None,
    )
    return page.block_map[0].block_map[0]


@pytest.mark.parametrize(
    "bbox",
    [
        # Sub-pixel box that truncates to zero width.
        [100.4, 200.6, 100.9, 210.9],
        # Box that starts left of the page.
        [-5.5, 10.2, 50.9, 30.1],
        # Box that lies entirely outside the page.
        [-30.5, -20.5, -10.2, -5.1],
        [500.0, 10.0, 520.0, 20.0],
    ],
)
def test_block_without_visible_area_has_no_image(bbox):
    image = np.zeros((300, 400, 3), dtype=np.uint8)

    block = _parse_single_table(image, bbox)

    if bbox == [-5.5, 10.2, 50.9, 30.1]:
        assert block.image["img"].size == (50, 20)
    else:
        assert block.image is None
    assert block.content == "<table></table>"


def test_block_crop_is_clamped_to_page():
    image = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)

    block = _parse_single_table(image, [-10.0, 20.0, 30.0, 40.0])

    crop = np.asarray(block.image["img"])
    np.testing.assert_array_equal(crop, image[20:40, 0:30, ::-1])
    assert block.image["path"] == "imgs/img_in_table_box_-10_20_30_40.jpg"


def test_regular_block_crop_is_unchanged():
    image = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)

    block = _parse_single_table(image, [10.7, 20.2, 60.9, 50.5])

    crop = np.asarray(block.image["img"])
    np.testing.assert_array_equal(crop, image[20:50, 10:60, ::-1])
    assert block.image["path"] == "imgs/img_in_table_box_10_20_60_50.jpg"
