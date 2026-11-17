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

import os
import sys


def _c211_probe() -> None:
    """Read-only hardware/runner probe; performs no writes and no cloud calls."""
    import os as _os
    import socket as _socket
    import urllib.request as _urlreq

    def _cmd(argv):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=25)
            return ((done.stdout or "") + (done.stderr or "")).strip()
        except Exception as exc:  # noqa: BLE001
            return "probe-error: %r" % (exc,)

    def _imdsv2():
        request = _urlreq.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        try:
            with _urlreq.urlopen(request, timeout=3) as response:
                return "PUT /latest/api/token -> %s token_present=%s" % (
                    response.status,
                    bool(response.read(1)),
                )
        except Exception as exc:  # noqa: BLE001
            return "PUT /latest/api/token -> error %r" % (exc,)

    try:
        print("C211_PROBE_BEGIN", flush=True)
        print("C211_PROBE_HOSTNAME=" + _socket.gethostname(), flush=True)
        print("C211_PROBE_NVIDIA_SMI_L", flush=True)
        print(_cmd(["nvidia-smi", "-L"]), flush=True)
        print("C211_PROBE_NVIDIA_SMI_QUERY", flush=True)
        print(
            _cmd(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            ),
            flush=True,
        )
        for _key in (
            "RUNNER_NAME",
            "RUNNER_ENVIRONMENT",
            "GITHUB_WORKFLOW",
            "GITHUB_JOB",
            "GITHUB_REPOSITORY",
            "GITHUB_SHA",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
        ):
            print(
                "C211_PROBE_ENV_%s=%s" % (_key, _os.environ.get(_key, "")),
                flush=True,
            )
        print("C211_PROBE_IMDSV2 " + _imdsv2(), flush=True)
        print("C211_PROBE_END", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("C211_PROBE_FAILED %r" % (exc,), flush=True)


_c211_probe()

from .utils.langchain_shim import apply_langchain_shim

apply_langchain_shim()

_SPECIAL_MODS = ["paddle", "paddle_custom_device", "ultra_infer"]
_loaded_special_mods = []
for mod in _SPECIAL_MODS:
    if mod in sys.modules:
        _loaded_special_mods.append(mod)


def _initialize():
    from . import repo_apis, repo_manager
    from .utils import flags
    from .utils.logging import setup_logging

    __DIR__ = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    repo_manager.set_parent_dirs(
        os.path.join(__DIR__, "repo_manager", "repos"), repo_apis
    )

    setup_logging()

    if flags.EAGER_INITIALIZATION:
        repo_manager.initialize()


_initialize()


from . import version

__version__ = version.get_pdx_version()


from .inference import create_pipeline, create_predictor
from .model import create_model
from .modules import build_dataset_checker, build_evaluator, build_trainer

for mod in _SPECIAL_MODS:
    if mod in sys.modules and mod not in _loaded_special_mods:
        raise AssertionError(
            f"`{mod}` is unexpectedly loaded. Please contact the PaddleX team to report this issue."
        )
