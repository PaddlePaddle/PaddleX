# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from six import text_type as _text_type
import argparse
import sys
import os
import os.path as osp


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_restful",
        "-sr",
        action="store_true",
        default=False,
        help="start paddlex restful server")
    parser.add_argument(
        "--port",
        "--pt",
        type=_text_type,
        default=None,
        help="set the port of restful server")
    parser.add_argument(
        "--workspace_dir",
        "--wd",
        type=_text_type,
        default=None,
        help="set the workspace dir of restful server")
    return parser

def main():
    if len(sys.argv) < 2:
        print("Use command 'paddlex_restful -h` to print the help information\n")
        return
    parser = arg_parser()
    args = parser.parse_args()

    if args.start_restful:
        import paddlex_restful as pdxr
        assert args.port is not None, "--port should be defined while start restful server"
        assert args.workspace_dir, "--workspace_dir should be define while start restful server"

        port = args.port
        workspace_dir = args.workspace_dir

        pdxr.restful.app.run(port, workspace_dir)


if __name__ == "__main__":
    main()
