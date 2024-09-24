# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import re
import sys

LICENSE_TEXT = """# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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
"""


def check(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    if not content.startswith(LICENSE_TEXT):
        print(f"License header missing in {file_path}")
        return False
    if "import paddle" in content or "from paddle import " in content:
        print(f"Please using `lazy_paddle` instead `paddle` when import in {file_path}")
        return False
    return True


def main():
    files = sys.argv[1:]
    all_files_valid = True
    for file in files:
        if not check(file):
            all_files_valid = False
    if not all_files_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
