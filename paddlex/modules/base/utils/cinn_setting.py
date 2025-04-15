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

from ....utils import logging

CINN_WHITELIST = [
    # TODO: update models based testing result
    "PP-LCNet_x0_25",
]


# TODO(gaotingquan): paddle v3.0.0 don't support enable CINN easily
def enable_cinn_backend():
    import paddle

    if not paddle.is_compiled_with_cinn():
        logging.debug(
            "Your paddle is not compiled with CINN, can not use CINN backend."
        )
        return

    # equivalent to `FLAGS_prim_all=1`
    paddle.base.core._set_prim_all_enabled(True)
    # equivalent to `FLAGS_prim_enable_dynamic=1`
    paddle.base.framework.set_flags({"FLAGS_prim_enable_dynamic": True})
    os.environ["FLAGS_prim_enable_dynamic"] = "1"
    # equivalent to `FLAGS_use_cinn=1`
    paddle.base.framework.set_flags({"FLAGS_use_cinn": True})
    os.environ["FLAGS_use_cinn"] = "1"
