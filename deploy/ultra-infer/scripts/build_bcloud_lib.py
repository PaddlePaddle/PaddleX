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
import shutil
import sys

dirname = sys.argv[1]
bc_dirname = sys.argv[2]

if os.path.exists(bc_dirname):
    raise Exception("Path {} is already exists.".format(bc_dirname))

os.makedirs(bc_dirname)

# copy include files
shutil.copytree(os.path.join(dirname, "include"), os.path.join(bc_dirname, "include"))

# copy libraries
shutil.copytree(os.path.join(dirname, "lib"), os.path.join(bc_dirname, "lib"))

third_libs = os.path.join(dirname, "third_libs")

for root, dirs, files in os.walk(third_libs):
    for f in files:
        if f.strip().count(".so") > 0 or f.strip() == "plugins.xml":
            full_path = os.path.join(root, f)
            shutil.copy(
                full_path, os.path.join(bc_dirname, "lib"), follow_symlinks=False
            )
