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
from paddlex.repo_manager import setup, get_all_supported_repo_names

if __name__ == '__main__':
    # Enable debug info
    os.environ['PADDLE_PDX_DEBUG'] = 'True'
    # Disable eager initialization
    os.environ['PADDLE_PDX_EAGER_INIT'] = 'False'

    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    repo_names = args.devkits
    if len(repo_names) == 0:
        repo_names = get_all_supported_repo_names()
    setup(
        repo_names=repo_names,
        reinstall=args.reinstall or None,
        no_deps=args.no_deps,
        platform=args.platform,
        update_repos=args.update_repos)
