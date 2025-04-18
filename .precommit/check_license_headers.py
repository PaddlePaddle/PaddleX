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

import re
import sys

YEAR_PATTERN = r"(?:20\d\d)"
LICENSE_HEADER_PATTERN = re.compile(
    re.escape(
        """# Copyright (c) <YEAR_PATTERN> PaddlePaddle Authors. All Rights Reserved.
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
    ).replace("<YEAR_PATTERN>", YEAR_PATTERN)
)


def check(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        contents = f.read()
    # Exclude shebang line
    if contents.startswith("#!"):
        contents = contents[contents.index("\n") + 1 :]
        if contents.startswith("\n"):
            contents = contents[1:]
    if not LICENSE_HEADER_PATTERN.match(contents):
        print(f"License header missing in `{file_path}`")
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
