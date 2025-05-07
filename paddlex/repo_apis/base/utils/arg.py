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
import shlex


class CLIArgument(object):
    """CLIArgument"""

    def __init__(self, key, *vals, quote=False, sep=" "):
        super().__init__()
        self.key = str(key)
        self.vals = [str(v) for v in vals]
        if quote and os.name != "posix":
            raise ValueError("`quote` cannot be True on non-POSIX compliant systems.")
        self.quote = quote
        self.sep = sep

    def __repr__(self):
        return self.sep.join(self.lst)

    @property
    def lst(self):
        """lst"""
        if self.quote:
            vals = [shlex.quote(val) for val in self.vals]
        else:
            vals = self.vals
        return [self.key, *vals]


def gather_opts_args(args, opts_key):
    """gather_opts_args"""

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
