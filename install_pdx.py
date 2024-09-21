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


import os
import argparse
from paddlex.repo_manager import setup, get_all_supported_repo_names

if __name__ == "__main__":
    # Enable debug info
    os.environ["PADDLE_PDX_DEBUG"] = "True"
    # Disable eager initialization
    os.environ["PADDLE_PDX_EAGER_INIT"] = "False"

    parser = argparse.ArgumentParser()
    parser.add_argument("devkits", nargs="*", default=[])
    parser.add_argument("--no_deps", action="store_true")
    parser.add_argument("--platform", type=str, default="github.com")
    parser.add_argument("--update_repos", action="store_true")
    parser.add_argument(
        "-y",
        "--yes",
        dest="reinstall",
        action="store_true",
        help="Whether to reinstall all packages.",
    )
    args = parser.parse_args()

    repo_names = args.devkits
    if len(repo_names) == 0:
        repo_names = get_all_supported_repo_names()
    setup(
        repo_names=repo_names,
        reinstall=args.reinstall or None,
        no_deps=args.no_deps,
        platform=args.platform,
        update_repos=args.update_repos,
    )
