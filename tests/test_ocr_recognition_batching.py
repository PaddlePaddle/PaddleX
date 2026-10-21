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

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytest


def _load_pipeline():
    path = (
        Path(__file__).resolve().parents[1]
        / "paddlex/inference/pipelines/ocr/pipeline.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_OCRPipeline"
    )
    cls.decorator_list = []
    namespace = dict(
        np=np,
        Any=Any,
        Dict=Dict,
        List=List,
        Optional=Optional,
        Union=Union,
        BasePipeline=object,
        HPIConfig=dict,
        PaddlePredictorOption=object,
        OCRResult=dict,
        convert_points_to_boxes=lambda polys: np.asarray(polys),
        cal_ocr_word_box=lambda text, poly, box: ([text], [poly]),
        rotate_image=lambda image, angle: np.rot90(image, k=angle // 90),
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_OCRPipeline"]


OCRPipeline = _load_pipeline()


class RecognitionModel:
    def __init__(self):
        self.calls = []

    def __call__(self, crops, return_word_box=False):
        self.calls.append([int(crop[0, 0, 0]) for crop in crops])
        for crop in crops:
            crop_id = int(crop[0, 0, 0])
            text = str(crop_id)
            yield {
                "rec_text": (text, [crop_id]) if return_word_box else text,
                "rec_score": 0.25 if crop_id == 2 else 0.9,
                "vis_font": "font-" + text,
            }


def _pipeline(cross_image, text_type, word_boxes, orientation, preprocessing):
    pipeline = object.__new__(OCRPipeline)
    pipeline.batch_recognition_across_images = cross_image
    pipeline.text_rec_score_thresh = 0
    pipeline.return_word_box = word_boxes
    pipeline.text_type = text_type
    images = [np.full((2, 2, 3), idx, dtype=np.uint8) for idx in range(4)]
    crops = {
        1: np.full((2, 8, 3), 1, dtype=np.uint8),
        2: np.full((4, 4, 3), 2, dtype=np.uint8),
        3: np.full((2, 4, 3), 3, dtype=np.uint8),
        4: np.full((4, 4, 3), 4, dtype=np.uint8),
        5: np.empty((0, 4, 3), dtype=np.uint8),
    }
    polygons = [[1, 2], [], [3, 4], [5]]
    batch = SimpleNamespace(
        instances=images,
        input_paths=["a", "b", "c", "d"],
        page_indexes=[0, 1, 2, 3],
    )
    pipeline.batch_sampler = lambda inputs: [batch]
    pipeline.img_reader = lambda inputs: inputs
    pipeline.get_model_settings = lambda *args: {
        "use_doc_preprocessor": preprocessing,
        "use_textline_orientation": orientation,
    }
    pipeline.check_model_settings_valid = lambda settings: True
    pipeline.get_text_det_params = lambda *args: {"thresh": 0.3}
    pipeline.doc_preprocessor_pipeline = lambda imgs, **kwargs: (
        {"output_img": img} for img in imgs
    )
    pipeline.text_det_model = lambda imgs, **kwargs: (
        {"dt_polys": polys} for polys in polygons
    )
    pipeline._sort_boxes = lambda polys: polys
    pipeline._crop_by_polys = lambda img, polys: (crops[idx] for idx in polys)
    pipeline.textline_orientation_model = lambda imgs: (
        {"class_ids": [idx % 2]} for idx in range(len(imgs))
    )
    pipeline.text_rec_model = RecognitionModel()
    return pipeline


@pytest.mark.parametrize("text_type", ["general", "seal"])
@pytest.mark.parametrize("word_boxes", [False, True])
@pytest.mark.parametrize("orientation", [False, True])
@pytest.mark.parametrize("preprocessing", [False, True])
@pytest.mark.parametrize("threshold", [0, 0.5])
def test_cross_image_recognition_preserves_result_mapping(
    text_type, word_boxes, orientation, preprocessing, threshold
):
    baseline = _pipeline(False, text_type, word_boxes, orientation, preprocessing)
    batched = _pipeline(True, text_type, word_boxes, orientation, preprocessing)
    expected = list(baseline.predict([], text_rec_score_thresh=threshold))
    actual = list(batched.predict([], text_rec_score_thresh=threshold))
    assert [item["input_path"] for item in actual] == ["a", "b", "c", "d"]
    for left, right in zip(actual, expected):
        assert left.keys() == right.keys()
        for key in left:
            if key == "doc_preprocessor_res":
                np.testing.assert_array_equal(
                    left[key]["output_img"], right[key]["output_img"]
                )
            else:
                np.testing.assert_equal(left[key], right[key])
    # Stable global aspect-ratio order: ties retain their crop IDs.
    assert batched.text_rec_model.calls == [[2, 4, 3, 1]]
    assert baseline.text_rec_model.calls == [[2, 1], [4, 3]]
    assert actual[1]["rec_texts"] == actual[3]["rec_texts"] == []


def test_no_recognition_call_for_empty_crops():
    pipeline = _pipeline(True, "general", False, False, False)
    assert pipeline._recognize_crops([], False) == []
    assert pipeline.text_rec_model.calls == []


def test_recognition_count_mismatch_is_reported():
    pipeline = _pipeline(True, "general", False, False, False)
    pipeline.text_rec_model = lambda crops, **kwargs: []
    with pytest.raises(ValueError, match="crop count"):
        pipeline._recognize_crops([np.ones((1, 2, 3))], False)
