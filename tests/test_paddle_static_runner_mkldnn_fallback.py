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

"""MKLDNN-to-paddle fallback of `PaddleStaticRunner`. Nothing is downloaded."""

import numpy as np
import pytest

from paddlex.inference.models.runners.paddle_static import runner as runner_module
from paddlex.inference.models.runners.paddle_static.runner import PaddleStaticRunner

ONEDNN_ERROR = (
    "(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support "
    "[pir::ArrayAttribute<pir::DoubleAttribute>]  (at /paddle/paddle/fluid/framework/"
    "new_executor/instruction/onednn/onednn_instruction.cc:116)\n"
)


class _Handle:
    def __init__(self, store):
        self._store = store

    def reshape(self, shape):
        pass

    def copy_from_cpu(self, arr):
        self._store["x"] = arr

    def copy_to_cpu(self):
        return self._store["x"] * self._store["factor"]


class _FakePredictor:
    """Stands in for `paddle.inference.Predictor`; `error` is raised by `run()`."""

    def __init__(self, run_mode, error=None):
        self.run_mode = run_mode
        self.error = error
        self.run_calls = 0
        # The output tells which predictor produced it.
        self._store = {"factor": 2.0 if run_mode == "paddle" else 3.0}

    def get_input_names(self):
        return ["x"]

    def get_output_names(self):
        return ["y"]

    def get_input_handle(self, name):
        return _Handle(self._store)

    def get_output_handle(self, name):
        return _Handle(self._store)

    def run(self):
        self.run_calls += 1
        if self.error is not None:
            raise self.error


def _make_runner(
    monkeypatch, *, run_mode, device_type="cpu", errors=(), requested=False
):
    """Build a runner whose `_create` returns fake predictors.

    `errors[i]` is the exception raised by the i-th created predictor (None: it works).
    `requested` tells whether the caller asked for `run_mode` or got it by default.
    """
    created = []

    def fake_create(self):
        error = errors[len(created)] if len(created) < len(errors) else None
        predictor = _FakePredictor(self._config["run_mode"], error)
        created.append(predictor)
        return predictor

    monkeypatch.setattr(
        runner_module,
        "resolve_paddle_static_engine_config",
        lambda model_name, config: {"run_mode": run_mode, "device_type": device_type},
    )
    monkeypatch.setattr(PaddleStaticRunner, "_create", fake_create)
    warnings = []
    monkeypatch.setattr(runner_module.logging, "warning", warnings.append)
    config = {"run_mode": run_mode} if requested else {"cpu_threads": 2}
    runner = PaddleStaticRunner("PP-OCRv5_mobile_det", "unused", "inference", config)
    return runner, created, warnings


X = [np.ones((1, 3, 4, 4), dtype=np.float32)]


def test_unimplemented_error_in_mkldnn_mode_falls_back_to_paddle(monkeypatch):
    runner, created, warnings = _make_runner(
        monkeypatch, run_mode="mkldnn", errors=[NotImplementedError(ONEDNN_ERROR)]
    )

    out = runner(X)

    assert [p.run_mode for p in created] == ["mkldnn", "paddle"]
    assert runner._config["run_mode"] == "paddle"
    assert runner.predictor is created[1]
    np.testing.assert_array_equal(out[0], X[0] * 2.0)  # from the `paddle` predictor
    assert len(warnings) == 1
    assert "PP-OCRv5_mobile_det" in warnings[0]
    assert "ConvertPirAttribute2RuntimeAttribute" in warnings[0]
    assert "\n" not in warnings[0]

    # Later calls use the rebuilt predictor directly: no new predictor, no new warning.
    runner(X)
    assert len(created) == 2
    assert created[1].run_calls == 2
    assert len(warnings) == 1


def test_requested_mkldnn_mode_is_not_changed(monkeypatch):
    runner, created, warnings = _make_runner(
        monkeypatch,
        run_mode="mkldnn",
        errors=[NotImplementedError(ONEDNN_ERROR)],
        requested=True,
    )

    with pytest.raises(
        NotImplementedError, match="ConvertPirAttribute2RuntimeAttribute"
    ):
        runner(X)

    assert len(created) == 1
    assert runner._config["run_mode"] == "mkldnn"
    assert warnings == []


def test_working_mkldnn_is_left_alone(monkeypatch):
    runner, created, warnings = _make_runner(monkeypatch, run_mode="mkldnn")

    out = runner(X)

    assert [p.run_mode for p in created] == ["mkldnn"]
    assert runner._config["run_mode"] == "mkldnn"
    np.testing.assert_array_equal(out[0], X[0] * 3.0)  # from the MKLDNN predictor
    assert warnings == []


def test_other_errors_in_mkldnn_mode_are_not_swallowed(monkeypatch):
    runner, created, warnings = _make_runner(
        monkeypatch, run_mode="mkldnn", errors=[RuntimeError("out of memory")]
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        runner(X)

    assert len(created) == 1
    assert runner._config["run_mode"] == "mkldnn"
    assert warnings == []


@pytest.mark.parametrize(
    "run_mode, device_type", [("paddle", "cpu"), ("paddle", "gpu"), ("trt_fp32", "gpu")]
)
def test_unimplemented_error_outside_mkldnn_mode_is_raised(
    monkeypatch, run_mode, device_type
):
    runner, created, warnings = _make_runner(
        monkeypatch,
        run_mode=run_mode,
        device_type=device_type,
        errors=[NotImplementedError(ONEDNN_ERROR)],
    )

    with pytest.raises(NotImplementedError):
        runner(X)

    assert len(created) == 1
    assert warnings == []


def test_fallback_is_tried_only_once(monkeypatch):
    error = NotImplementedError("(Unimplemented) some kernel is missing")
    runner, created, warnings = _make_runner(
        monkeypatch, run_mode="mkldnn", errors=[error, error]
    )

    # The `paddle` predictor fails too: the error reaches the caller.
    with pytest.raises(NotImplementedError, match="some kernel is missing"):
        runner(X)
    assert len(created) == 2

    # No rebuild on every later call.
    with pytest.raises(NotImplementedError):
        runner(X)
    assert len(created) == 2
    assert len(warnings) == 1


def test_exported_model_with_upsample_runs_in_default_cpu_mode(tmp_path):
    """A real predictor, no download. PaddlePaddle 3.3.0, 3.3.1 and 3.4.0 cannot
    run the upsample through oneDNN (PaddlePaddle/Paddle#77340)."""
    paddle = pytest.importorskip("paddle")

    class Net(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.conv = paddle.nn.Conv2D(3, 8, 3, padding=1)
            self.bn = paddle.nn.BatchNorm2D(8)

        def forward(self, x):
            y = paddle.nn.functional.relu(self.bn(self.conv(x)))
            return paddle.nn.functional.interpolate(y, scale_factor=2, mode="nearest")

    paddle.seed(0)
    net = Net()
    net.eval()
    spec = [paddle.static.InputSpec([None, 3, 16, 16], "float32", "x")]
    paddle.jit.save(net, str(tmp_path / "inference"), input_spec=spec)
    x = [np.random.RandomState(0).rand(1, 3, 16, 16).astype(np.float32)]

    default = PaddleStaticRunner("tiny_upsample", tmp_path, "inference", {})
    plain = PaddleStaticRunner(
        "tiny_upsample", tmp_path, "inference", {"run_mode": "paddle"}
    )

    out = default(x)[0]
    assert out.shape == (1, 8, 32, 32)
    np.testing.assert_allclose(out, plain(x)[0], atol=1e-5)
