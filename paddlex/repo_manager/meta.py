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

__all__ = ["get_all_repo_names"]

REPO_DOWNLOAD_BASE = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/repos/"
)

REPO_NAMES = [
    "PaddleClas",
    "PaddleOCR",
    "PaddleDetection",
    "PaddleSeg",
    "PaddleNLP",
    "PaddleTS",
    "Paddle3D",
    "PaddleVideo",
]

REPO_META = {
    "PaddleSeg": {
        "git_path": "/PaddlePaddle/PaddleSeg.git",
        "platform": "github",
        "branch": "develop",
        "lib_name": "paddleseg",
        "install_pkg": True,
        "pkg_name": "paddleseg",
        "pdx_pkg_name": "PaddleSeg_api",
        "editable": False,
        "extra_pkgs": ["Matting/requirements.txt"],
        "path_env": "PADDLE_PDX_PADDLESEG_PATH",
    },
    "PaddleClas": {
        "git_path": "/PaddlePaddle/PaddleClas.git",
        "platform": "github",
        "branch": "develop",
        "lib_name": "paddleclas",
        "install_pkg": True,
        "pkg_name": "paddleclas",
        "pdx_pkg_name": "PaddleClas_api",
        # PaddleClas must be installed in non-editable mode, otherwise it throws
        # an Import error.
        "editable": False,
        "path_env": "PADDLE_PDX_PADDLECLAS_PATH",
    },
    "PaddleDetection": {
        "git_path": "/PaddlePaddle/PaddleDetection.git",
        "platform": "github",
        "branch": "develop",
        "lib_name": "ppdet",
        "install_pkg": True,
        "pkg_name": "paddledet",
        "pdx_pkg_name": "PaddleDetection_api",
        "editable": False,
        "path_env": "PADDLE_PDX_PADDLEDETECTION_PATH",
    },
    "PaddleOCR": {
        "git_path": "/PaddlePaddle/PaddleOCR.git",
        "platform": "github",
        "branch": "main",
        "lib_name": "paddleocr",
        "install_pkg": False,
        "pdx_pkg_name": "PaddleOCR_api",
        "extra_pkgs": [
            "ppstructure/kie/requirements.txt",
            "docs/algorithm/formula_recognition/requirements.txt",
        ],
        "path_env": "PADDLE_PDX_PADDLEOCR_PATH",
        "requires": ["PaddleNLP"],
    },
    "PaddleTS": {
        "git_path": "/PaddlePaddle/PaddleTS.git",
        "platform": "github",
        "branch": "release_v1.1",
        "lib_name": "paddlets",
        "install_pkg": True,
        "pkg_name": "paddlets",
        "pdx_pkg_name": "PaddleTS_api",
        "editable": False,
        "path_env": "PADDLE_PDX_PADDLETS_PATH",
        "pdx_pkg_deps": ["pandas", "ruamel.yaml"],
    },
    "PaddleNLP": {
        "git_path": "/PaddlePaddle/PaddleNLP.git",
        "platform": "github",
        "branch": "release/2.9",
        "lib_name": "paddlenlp",
        "install_pkg": True,
        "pkg_name": "paddlenlp",
        "pdx_pkg_name": "PaddleNLP_api",
        "editable": False,
        "path_env": "PADDLE_PDX_PADDLENLP_PATH",
    },
    "PaddleSpeech": {
        "git_path": "/PaddlePaddle/PaddleSpeech.git",
        "platform": "github",
        "branch": "develop",
        "lib_name": "paddlespeech",
        "install_pkg": True,
        "pkg_name": "paddlespeech",
        "pdx_pkg_name": "PaddleSpeech_api",
        "editable": False,
        "path_env": "PADDLE_PDX_PADDLESPEECH_PATH",
        "requires": ["PaddleNLP"],
    },
    "PARL": {
        "git_path": "/PaddlePaddle/PARL.git",
        "platform": "github",
        "branch": "develop",
        "lib_name": "parl",
        "install_pkg": True,
        "pkg_name": "parl",
        "pdx_pkg_name": "PARL_api",
        "editable": False,
        "path_env": "PADDLE_PDX_PARL_PATH",
    },
    "PaddleMIX": {
        "git_path": "/PaddlePaddle/PaddleMIX.git",
        "platform": "github",
        "branch": "develop",
        "lib_name": "paddlemix",
        "pkg_name": "paddlemix",
        "pdx_pkg_name": "PaddleMIX_api",
        "editable": True,
        "extra_pkgs": [("ppdiffusers", "ppdiffusers", None, True)],
        "path_env": "PADDLE_PDX_PADDLEMIX_PATH",
        "requires": ["PaddleNLP"],
    },
    "Paddle3D": {
        "git_path": "/PaddlePaddle/Paddle3D.git",
        "platform": "github",
        "branch": "develop",
        "lib_name": "paddle3d",
        "install_pkg": True,
        "pkg_name": "paddle3d",
        "pdx_pkg_name": "Paddle3D_api",
        "editable": False,
        "path_env": "PADDLE_PDX_PADDLE3D_PATH",
        "requires": ["PaddleSeg", "PaddleDetection"],
        "main_reqs_file": "requirements_pdx.txt",
    },
    "PaddleVideo": {
        "git_path": "/PaddlePaddle/PaddleVideo.git",
        "platform": "github",
        "branch": "develop",
        "lib_name": "ppvideo",
        "install_pkg": True,
        "pkg_name": "paddlevideo",
        "pdx_pkg_name": "PaddleVideo_api",
        "editable": False,
        "main_reqs_file": "requirements_paddlex.txt",
        "path_env": "PADDLE_PDX_PADDLEVIDEO_PATH",
    },
}


def get_repo_meta(repo_name):
    """get_repo_meta"""
    return REPO_META[repo_name]


def get_all_repo_names():
    """get_all_repo_names"""
    return REPO_NAMES
