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
import shlex


class CLIArgument(object):
    """ CLIArgument """

    def __init__(self, key, *vals, quote=False, sep=' '):
        super().__init__()
        self.key = str(key)
        self.vals = [str(v) for v in vals]
        if quote and os.name != 'posix':
            raise ValueError(
                "`quote` cannot be True on non-POSIX compliant systems.")
        self.quote = quote
        self.sep = sep

    def __repr__(self):
        return self.sep.join(self.lst)

    @property
    def lst(self):
        """ lst """
        if self.quote:
            vals = [shlex.quote(val) for val in self.vals]
        else:
            vals = self.vals
        return [self.key, *vals]


def gather_opts_args(args, opts_key):
    """ gather_opts_args """

    def _is_opts_arg(arg):
        return arg.key == opts_key

    args = sorted(args, key=_is_opts_arg)
    idx = None
    for i, arg in enumerate(args):
        if _is_opts_arg(arg):
            idx = i
            break
    if idx is not None:
        opts_args = args[idx:]
        args = args[:idx]
        all_vals = []
        for arg in opts_args:
            all_vals.extend(arg.vals)
        args.append(CLIArgument(opts_key, *all_vals))
    return args
