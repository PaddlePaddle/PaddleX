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


def process_paddle_lite(paddle_lite_so_path):
    if platform.system().lower() != "linux":
        return
    rpaths = ["$ORIGIN", "$ORIGIN/mklml/lib/"]
    patchelf_exe = os.getenv("PATCHELF_EXE", "patchelf")
    for root, dirs, files in os.walk(paddle_lite_so_path):
        for lib in files:
            if ".so" in lib:
                paddle_lite_so_file = os.path.join(root, lib)
                command = "{} --set-rpath '{}' {}".format(
                    patchelf_exe, ":".join(rpaths), paddle_lite_so_file
                )
                if platform.machine() != "sw_64" and platform.machine() != "mips64":
                    assert (
                        os.system(command) == 0
                    ), "patchelf {} failed, the command: {}".format(
                        paddle_lite_so_file, command
                    )


if __name__ == "__main__":
    process_paddle_lite(sys.argv[1])
