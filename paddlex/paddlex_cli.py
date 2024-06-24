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
import os
import argparse
import textwrap
from types import SimpleNamespace

from .pipelines import build_pipeline, BasePipeline
from .repo_manager import setup, get_all_supported_repo_names
from .utils import logging


def args_cfg():
    """parse cli arguments
    """

    def str2bool(v):
        """convert str to bool type
        """
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()

    ################# install pdx #################
    parser.add_argument(
        '--install', action='store_true', default=False, help="")
    parser.add_argument('devkits', nargs='*', default=[])
    parser.add_argument('--no_deps', action='store_true')
    parser.add_argument('--platform', type=str, default='github.com')
    parser.add_argument('--update_repos', action='store_true')
    parser.add_argument(
        '-y',
        '--yes',
        dest='reinstall',
        action='store_true',
        help="Whether to reinstall all packages.")

    ################# pipeline predict #################
    parser.add_argument('--predict', action='store_true', default=True, help="")
    parser.add_argument('--pipeline', type=str, help="")
    parser.add_argument('--model', nargs='+', help="")
    parser.add_argument('--input', type=str, help="")
    parser.add_argument('--output', type=str, default="./", help="")
    parser.add_argument('--device', type=str, default='gpu:0', help="")

    return parser.parse_args()


def install(args):
    """install paddlex
    """
    # Enable debug info
    os.environ['PADDLE_PDX_DEBUG'] = 'True'
    # Disable eager initialization
    os.environ['PADDLE_PDX_EAGER_INIT'] = 'False'

    repo_names = args.devkits
    if len(repo_names) == 0:
        repo_names = get_all_supported_repo_names()
    setup(
        repo_names=repo_names,
        reinstall=args.reinstall or None,
        no_deps=args.no_deps,
        platform=args.platform,
        update_repos=args.update_repos)
    return


def pipeline_predict(pipeline, model_name_list, input_path, output, device):
    """pipeline predict
    """
    pipeline = build_pipeline(pipeline, model_name_list, output, device)
    pipeline.predict({"input_path": input_path})


# for CLI
def main():
    """API for commad line
    """
    args = args_cfg()
    if args.install:
        install(args)
    else:
        return pipeline_predict(args.pipeline, args.model, args.input,
                                args.output, args.device)
