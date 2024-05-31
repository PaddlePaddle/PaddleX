# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: PaddlePaddle Authors
"""

__all__ = ['get_all_repo_names']

# REPO_NAMES = [
#     'PaddleSeg', 'Paddle3D', 'PaddleClas', 'PaddleDetection', 'PaddleOCR',
#     'PaddleTS', 'PaddleNLP', 'PaddleSpeech', 'PARL', 'PaddleMIX']

REPO_NAMES = [
    'PaddleClas', 'PaddleOCR', 'PaddleDetection', 'PaddleSeg', 'PaddleNLP',
    'PaddleTS'
]

REPO_META = {
    'PaddleSeg': {
        'repo_url': '/PaddlePaddle/PaddleSeg.git',
        'branch': 'develop',
        'pkg_name': 'paddleseg',
        'lib_name': 'paddleseg',
        'pdx_pkg_name': 'PaddleSeg_api',
        'editable': False,
        'extra_req_files': ['Matting/requirements.txt'],
        'path_env': 'PADDLE_PDX_PADDLESEG_PATH',
    },
    'Paddle3D': {
        'repo_url': '/PaddlePaddle/Paddle3D.git',
        'branch': 'develop',
        'pkg_name': 'paddle3d',
        'lib_name': 'paddle3d',
        'pdx_pkg_name': 'Paddle3D_api',
        'editable': False,
        'path_env': 'PADDLE_PDX_PADDLE3D_PATH',
        'requires': ['PaddleSeg', 'PaddleDetection'],
        'pdx_pkg_deps': ['nuscenes-devkit', 'pyquaternion'],
    },
    'PaddleClas': {
        'repo_url': '/PaddlePaddle/PaddleClas.git',
        'branch': 'develop',
        'pkg_name': 'paddleclas',
        'lib_name': 'paddleclas',
        'pdx_pkg_name': 'PaddleClas_api',
        # PaddleClas must be installed in non-editable mode, otherwise it throws
        # an Import error.
        'editable': False,
        'path_env': 'PADDLE_PDX_PADDLECLAS_PATH',
    },
    'PaddleDetection': {
        'repo_url': '/PaddlePaddle/PaddleDetection.git',
        'branch': 'develop',
        'pkg_name': 'paddledet',
        'lib_name': 'ppdet',
        'pdx_pkg_name': 'PaddleDetection_api',
        'editable': False,
        'path_env': 'PADDLE_PDX_PADDLEDETECTION_PATH',
    },
    'PaddleOCR': {
        'repo_url': '/PaddlePaddle/PaddleOCR.git',
        'platform': 'github',
        'branch': 'release/2.6.1',
        'pkg_name': 'paddleocr',
        'lib_name': 'paddleocr',
        'pdx_pkg_name': 'PaddleOCR_api',
        'editable': False,
        'extra_req_files': ['ppstructure/kie/requirements.txt'],
        'path_env': 'PADDLE_PDX_PADDLEOCR_PATH',
        'requires': ['PaddleNLP'],
    },
    'PaddleTS': {
        'repo_url': '/PaddlePaddle/PaddleTS.git',
        'branch': 'release_v1.1',
        'pkg_name': 'paddlets',
        'lib_name': 'paddlets',
        'pdx_pkg_name': 'PaddleTS_api',
        'editable': False,
        'path_env': 'PADDLE_PDX_PADDLETS_PATH',
        'pdx_pkg_deps': ['pandas', 'ruamel.yaml'],
    },
    'PaddleNLP': {
        'repo_url': '/PaddlePaddle/PaddleNLP.git',
        'branch': 'develop',
        'pkg_name': 'paddlenlp',
        'lib_name': 'paddlenlp',
        'pdx_pkg_name': 'PaddleNLP_api',
        'editable': False,
        'path_env': 'PADDLE_PDX_PADDLENLP_PATH',
    },
    'PaddleSpeech': {
        'repo_url': '/PaddlePaddle/PaddleSpeech.git',
        'branch': 'develop',
        'pkg_name': 'paddlespeech',
        'lib_name': 'paddlespeech',
        'pdx_pkg_name': 'PaddleSpeech_api',
        'editable': False,
        'path_env': 'PADDLE_PDX_PADDLESPEECH_PATH',
        'requires': ['PaddleNLP'],
    },
    'PARL': {
        'repo_url': '/PaddlePaddle/PARL.git',
        'branch': 'develop',
        'pkg_name': 'parl',
        'lib_name': 'parl',
        'pdx_pkg_name': 'PARL_api',
        'editable': False,
        'path_env': 'PADDLE_PDX_PARL_PATH',
    },
    'PaddleMIX': {
        'repo_url': '/PaddlePaddle/PaddleMIX.git',
        'branch': 'develop',
        'pkg_name': 'paddlemix',
        'lib_name': 'paddlemix',
        'pdx_pkg_name': 'PaddleMIX_api',
        'editable': True,
        'extra_editable': 'ppdiffusers',
        'path_env': 'PADDLE_PDX_PADDLEMIX_PATH',
        'requires': ['PaddleNLP'],
    },
}


def get_repo_meta(repo_name):
    """ get_repo_meta """
    return REPO_META[repo_name]


def get_all_repo_names():
    """ get_all_repo_names """
    return REPO_NAMES
