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
from types import MethodType, SimpleNamespace
from typing import List, Union

import numpy as np
import pytest


def _load_process():
    path = (
        Path(__file__).resolve().parents[1]
        / "paddlex/inference/models/text_detection/predictor.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TextDetRunnerPredictor"
    )
    process = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "process"
    )
    namespace = {"np": np, "List": List, "Union": Union}
    exec(
        compile(ast.Module(body=[process], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["process"]


PROCESS = _load_process()


def _predictor():
    runner_calls, post_shapes, params = [], [], []

    def resize(imgs, **kwargs):
        params.append(kwargs)
        return imgs, [img.shape[:2] for img in imgs]

    def runner(x):
        batch = x[0]
        runner_calls.append([int(img[0, 0, 0]) for img in batch])
        return batch

    def post_op(preds, shapes, **kwargs):
        post_shapes.extend(shapes)
        params.append(kwargs)
        polys, scores = [], []
        for pred, shape in zip(preds, shapes):
            image_id = int(pred[0, 0, 0])
            polys.append(np.asarray([[image_id, shape[0], shape[1]]]))
            scores.append([image_id / 10])
        return polys, scores

    model = SimpleNamespace(
        limit_side_len=960,
        limit_type="max",
        max_side_limit=4000,
        thresh=0.3,
        box_thresh=0.6,
        unclip_ratio=2.0,
        pre_tfs={
            "Read": lambda imgs: imgs,
            "Resize": resize,
            "Normalize": lambda imgs: [img.astype(np.float32) for img in imgs],
            "ToCHW": lambda imgs: [img.transpose(2, 0, 1) for img in imgs],
            "ToBatch": lambda imgs: [np.stack(imgs)],
        },
        runner=runner,
        post_op=post_op,
        runner_calls=runner_calls,
        post_shapes=post_shapes,
        params=params,
    )
    model.process = MethodType(PROCESS, model)
    return model


def _batch(images):
    return SimpleNamespace(
        instances=images,
        input_paths=[f"image-{int(img[0, 0, 0])}" for img in images],
        page_indexes=[int(img[0, 0, 0]) for img in images],
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    "sizes", [[(4, 8), (8, 4), (4, 8)], [(4, 8)] * 3, [(4, 8), (8, 4), (8, 8)]]
)
def test_shape_buckets_restore_detection_order(dtype, sizes):
    imgs = [
        np.full((*shape, 3), idx + 1, dtype=dtype) for idx, shape in enumerate(sizes)
    ]
    baseline = _predictor()
    expected = [baseline.process(_batch([img])) for img in imgs]
    bucketed = _predictor()
    actual = bucketed.process(_batch(imgs), batch_by_shape=True)
    for idx, reference in enumerate(expected):
        for key in ("input_path", "page_index", "input_img", "dt_polys", "dt_scores"):
            np.testing.assert_equal(actual[key][idx], reference[key][0])
    expected_buckets = {}
    for idx, size in enumerate(sizes):
        expected_buckets.setdefault(size, []).append(idx + 1)
    assert bucketed.runner_calls == list(expected_buckets.values())
    expected_shapes = [
        sizes[idx - 1] for ids in expected_buckets.values() for idx in ids
    ]
    assert bucketed.post_shapes == expected_shapes


def test_disabled_option_keeps_existing_runner_batch():
    model = _predictor()
    imgs = [np.full((4, 8, 3), idx, dtype=np.uint8) for idx in (1, 2, 3)]
    model.process(_batch(imgs))
    assert model.runner_calls == [[1, 2, 3]]


def test_prediction_overrides_reach_resize_and_postprocess():
    model = _predictor()
    model.process(
        _batch([np.ones((4, 8, 3), dtype=np.uint8)]),
        batch_by_shape=True,
        limit_side_len=128,
        limit_type="min",
        max_side_limit=1024,
        thresh=0.4,
        box_thresh=0.7,
        unclip_ratio=1.5,
    )
    assert model.params == [
        {"limit_side_len": 128, "limit_type": "min", "max_side_limit": 1024},
        {"thresh": 0.4, "box_thresh": 0.7, "unclip_ratio": 1.5},
    ]


def test_result_count_mismatch_is_reported():
    model = _predictor()
    model.post_op = lambda *args, **kwargs: ([], [])
    with pytest.raises(ValueError, match="image count"):
        model.process(_batch([np.ones((4, 8, 3))]), batch_by_shape=True)
