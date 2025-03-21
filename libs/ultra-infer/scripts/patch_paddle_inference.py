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
import platform
import sys


def process_paddle_inference(paddle_inference_so_file, paddle_inference_version):
    if platform.system().lower() != "linux":
        return
    rpaths = [
        "$ORIGIN",
        "$ORIGIN/../../third_party/install/mklml/lib/",
        "$ORIGIN/../../third_party/install/xpu/lib/",
        "$ORIGIN/../../third_party/install/fdmodel/lib/",
        "$ORIGIN/../../../tensorrt/lib/",
    ]
    version_major = int(paddle_inference_version.split(".")[0])
    if paddle_inference_version != "0.0.0" and version_major < 2:
        raise ValueError("Invalid Paddle Inference version")
    if version_major == 2:
        rpaths.append("$ORIGIN/../../third_party/install/mkldnn/lib/")
    else:
        rpaths.append("$ORIGIN/../../third_party/install/onednn/lib/")

    patchelf_exe = os.getenv("PATCHELF_EXE", "patchelf")
    command = "{} --force-rpath --set-rpath '{}' {}".format(
        patchelf_exe, ":".join(rpaths), paddle_inference_so_file
    )
    if platform.machine() != "sw_64" and platform.machine() != "mips64":
        assert os.system(command) == 0, "patchelf {} failed, the command: {}".format(
            paddle_inference_so_file, command
        )


if __name__ == "__main__":
    process_paddle_inference(sys.argv[1], sys.argv[2])
