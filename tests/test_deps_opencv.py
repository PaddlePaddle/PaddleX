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

import importlib.machinery
import importlib.metadata
import importlib.util

import pytest

from paddlex.utils import deps

_OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)


@pytest.fixture(autouse=True)
def _clear_dep_cache():
    deps.is_dep_available.cache_clear()
    yield
    deps.is_dep_available.cache_clear()


def _simulate_opencv(monkeypatch, installed_distribution):
    """Simulate an environment where only `installed_distribution` provides `cv2`.

    `installed_distribution=None` simulates an environment without `cv2`.
    """
    real_find_spec = importlib.util.find_spec
    real_version = importlib.metadata.version

    def fake_find_spec(name, *args, **kwargs):
        if name == "cv2":
            if installed_distribution is None:
                return None
            return importlib.machinery.ModuleSpec("cv2", None)
        return real_find_spec(name, *args, **kwargs)

    def fake_version(name):
        if name in _OPENCV_DISTRIBUTIONS:
            if name == installed_distribution:
                return "4.10.0.84"
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(importlib.metadata, "version", fake_version)


@pytest.mark.parametrize("distribution", _OPENCV_DISTRIBUTIONS)
def test_opencv_available_with_any_cv2_distribution(monkeypatch, distribution):
    _simulate_opencv(monkeypatch, distribution)

    assert deps.is_dep_available("opencv-contrib-python")


def test_opencv_unavailable_without_cv2(monkeypatch):
    _simulate_opencv(monkeypatch, None)

    assert not deps.is_dep_available("opencv-contrib-python")


def test_require_deps_accepts_headless_opencv(monkeypatch):
    _simulate_opencv(monkeypatch, "opencv-python-headless")

    deps.require_deps("opencv-contrib-python")


def test_require_deps_rejects_missing_opencv(monkeypatch):
    _simulate_opencv(monkeypatch, None)

    with pytest.raises(deps.DependencyError):
        deps.require_deps("opencv-contrib-python")
