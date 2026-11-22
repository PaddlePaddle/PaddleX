# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2] / "paddlex/inference"


def load_method(path, class_name, name, **scope):
    tree = ast.parse((ROOT / path).read_text())
    owner = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    fn = next(
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    fn.decorator_list = []
    code = compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec")
    exec(code, scope)
    return scope[name]


def test_submit_each_crop_before_encoding_the_next_and_cancel_on_error():
    run = load_method(
        "models/doc_vlm/predictor.py",
        "DocVLMGenAIClientPredictor",
        "_genai_client_process",
    )
    pending = Mock()
    pending.done.return_value = False
    sent = []

    def encode(client, data, *args):
        assert len(data) == 1
        if data[0] == 2:
            assert sent == [1]
            raise ValueError("encode failed")
        return [(data[0], {})]

    def send(messages, **kwargs):
        sent.append(messages)
        return pending

    obj = SimpleNamespace(
        genai_client=SimpleNamespace(
            backend="vllm-server", create_chat_completion=send
        ),
        _doc_vlm_genai_build_request_specs=encode,
    )
    with pytest.raises(ValueError, match="encode failed"):
        run(obj, [1, 2], *([None] * 7))
    pending.cancel.assert_called_once_with()


@pytest.mark.parametrize("remote", [True, False])
def test_pixel_groups_only_overlap_for_remote_models(remote):
    run = load_method(
        "pipelines/paddleocr_vl/pipeline.py",
        "_PaddleOCRVLPipeline",
        "_paddleocr_vl_run_vl_recognition_batches",
        ThreadPoolExecutor=ThreadPoolExecutor,
    )
    gate = Barrier(2)

    def predict(requests, **kwargs):
        if remote:
            gate.wait(timeout=2)
        return iter([kwargs["skip_special_tokens"]])

    model = SimpleNamespace(model_name="PaddleOCR-VL-1.8-0.9B", predict=predict)
    obj = SimpleNamespace(_vl_rec_uses_server=remote, vl_rec_model=model)
    groups = {
        key: {"images": ["image"], "queries": ["query"]}
        for key in [(1, 2), (1, 2, "table")]
    }
    run(obj, groups, False, {})
    assert groups[(1, 2)]["vlm_results"] == [True]
    assert groups[(1, 2, "table")]["vlm_results"] == [False]


@pytest.mark.parametrize("style", ["legacy", "explicit", "local"])
def test_pipeline_uses_the_resolved_predictor(style):
    class GenAIClientPredictor:
        pass

    class Base:
        def __init__(self, **kwargs):
            pass

        def create_model(self, config):
            assert config == vlm_config
            return GenAIClientPredictor() if style != "local" else object()

    vlm_config = {
        "legacy": {"genai_config": {"backend": "vllm-server"}},
        "explicit": {
            "engine": "genai_client",
            "engine_config": {"backend": "vllm-server"},
        },
        "local": {"engine": "transformers"},
    }[style]
    # Compile the real constructor inside a class to retain its super() binding.
    path = ROOT / "pipelines/paddleocr_vl/pipeline.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "_PaddleOCRVLPipeline"
    )
    ctor = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.keywords = []
    cls.decorator_list = []
    cls.body = [ctor]
    scope = dict(
        Base=Base,
        GenAIClientPredictor=GenAIClientPredictor,
        Dict=Dict,
        Any=Any,
        Optional=Optional,
        Union=Union,
        HPIConfig=object,
        PaddlePredictorOption=object,
        ImageBatchSampler=Mock(),
        ReadImage=Mock(),
        CropByBoxes=Mock(),
        DEFAULT_CONF_THRESH=0,
        DEFAULT_MIN_ROLL=0,
        DEFAULT_MIN_SHEAR=0,
        DEFAULT_MIN_KEYSTONE=0,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    pipeline = scope[cls.name](
        {
            "use_doc_preprocessor": False,
            "use_layout_detection": False,
            "SubModules": {"VLRecognition": vlm_config},
        }
    )
    assert pipeline._vl_rec_uses_server == (style != "local")


def _run_queue_case(process, *, remote=True):
    predict = load_method(
        "pipelines/paddleocr_vl/pipeline.py",
        "_PaddleOCRVLPipeline",
        "predict",
        List=List,
        Union=Union,
        Optional=Optional,
        Tuple=Tuple,
        np=SimpleNamespace(ndarray=object),
        PaddleOCRVLResult=lambda result: result,
        queue=queue,
        threading=threading,
        time=time,
        chain=chain,
        Future=Future,
        ThreadPoolExecutor=ThreadPoolExecutor,
        logging=Mock(),
        MAX_NUM_VLM_BATCHES_IN_FLIGHT=2,
        MAX_NUM_PAGES_PER_VLM_BATCH=1,
        _is_uniform_image=lambda _: False,
    )

    images = [SimpleNamespace(shape=(64, 128), number=i) for i in range(3)]
    batches = [
        SimpleNamespace(
            instances=[im],
            input_paths=[str(i)],
            page_indexes=[i],
            page_counts=[len(images)],
        )
        for i, im in enumerate(images)
    ]

    class Batch(SimpleNamespace):
        def __len__(self):
            return len(self.instances)

    class Sampler:
        batch_size = 1

        def __call__(self, _):
            yield from (Batch(**vars(b)) for b in batches)

    model_settings = dict(
        use_layout_detection=False,
        use_doc_preprocessor=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        use_ocr_for_image_block=False,
        merge_layout_blocks=False,
        use_layout_deform=False,
    )
    pipeline = SimpleNamespace(
        get_model_settings=lambda *a, **k: model_settings,
        check_model_settings_valid=lambda _: True,
        use_queues=True,
        _vl_rec_uses_server=remote,
        batch_sampler=Sampler(),
        img_reader=lambda x: x,
        vl_rec_model=SimpleNamespace(batch_sampler=SimpleNamespace(batch_size=1)),
        layout_deform_conf_thresh=0,
        layout_deform_min_roll=0,
        layout_deform_min_shear=0,
        layout_deform_min_keystone=0,
        get_layout_parsing_results=process,
    )
    return predict(pipeline, images)


def test_queued_remote_batches_overlap_but_return_in_input_order():
    later_done = threading.Event()
    completed = []

    def process(images, **kwargs):
        index = images[0].number
        if index == 0:
            assert later_done.wait(3), "the next batch did not run concurrently"
        completed.append(index)
        if index == 1:
            later_done.set()
        return [[index]], [[]], [[]], [[]]

    results = list(_run_queue_case(process))
    assert [page["page_index"] for page in results] == [0, 1, 2]
    assert completed == [1, 0, 2]


def test_queued_remote_worker_errors_reach_the_caller():
    def fail(**kwargs):
        raise ValueError("remote service failed")

    with pytest.raises(RuntimeError, match="remote service failed"):
        list(_run_queue_case(fail))


def test_queued_local_predictor_stays_serial():
    thread_ids = []

    def process(images, **kwargs):
        thread_ids.append(threading.get_ident())
        return [[images[0].number]], [[]], [[]], [[]]

    assert [r["page_index"] for r in _run_queue_case(process, remote=False)] == [
        0,
        1,
        2,
    ]
    assert len(set(thread_ids)) == 1
