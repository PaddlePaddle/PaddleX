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
import textwrap
from types import SimpleNamespace
from prettytable import PrettyTable

from .pipelines import build_pipeline, BasePipeline
from .repo_manager import setup, get_all_supported_repo_names
from .utils import logging


def args_cfg():
    """parse cli arguments
    """

    def str2bool(v):
        """convert str to bool type
        """
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()

    ################# install pdx #################
    parser.add_argument(
        '--install', action='store_true', default=False, help="")
    parser.add_argument('devkits', nargs='*', default=[])
    parser.add_argument('--no_deps', action='store_true')
    parser.add_argument('--platform', type=str, default='github.com')
    parser.add_argument('--update_repos', action='store_true')
    parser.add_argument(
        '-y',
        '--yes',
        dest='reinstall',
        action='store_true',
        help="Whether to reinstall all packages.")

    ################# pipeline predict #################
    parser.add_argument('--predict', action='store_true', default=True, help="")
    parser.add_argument('--pipeline', type=str, help="")
    parser.add_argument('--model', nargs='+', help="")
    parser.add_argument('--input', type=str, help="")
    parser.add_argument('--output', type=str, help="")
    parser.add_argument('--device', type=str, default='gpu:0', help="")

    return parser.parse_args()


def print_info():
    """Print list of supported models in formatted.
    """
    try:
        sz = os.get_terminal_size()
        total_width = sz.columns
        first_width = 30
        second_width = total_width - first_width if total_width > 50 else 10
    except OSError:
        total_width = 100
        second_width = 100
    total_width -= 4

    pipeline_table = PrettyTable()
    pipeline_table.field_names = ["Pipelines"]
    pipeline_table.add_row(
        [textwrap.fill(
            ",  ".join(BasePipeline.all()), width=total_width)])

    table_width = len(str(pipeline_table).split("\n")[0])

    logging.info("{}".format("-" * table_width))
    logging.info("PaddleX".center(table_width))
    logging.info(pipeline_table)
    logging.info("Powered by PaddlePaddle!".rjust(table_width))
    logging.info("{}".format("-" * table_width))


def install(args):
    """install paddlex
    """
    # Enable debug info
    os.environ['PADDLE_PDX_DEBUG'] = 'True'
    # Disable eager initialization
    os.environ['PADDLE_PDX_EAGER_INIT'] = 'False'

    repo_names = args.devkits
    if len(repo_names) == 0:
        repo_names = get_all_supported_repo_names()
    setup(
        repo_names=repo_names,
        reinstall=args.reinstall or None,
        no_deps=args.no_deps,
        platform=args.platform,
        update_repos=args.update_repos)
    return


def pipeline_predict(pipeline, model_name_list, input_path, output_dir, device):
    pipeline = build_pipeline(pipeline, model_name_list, output_dir, device)
    pipeline.predict(input_path)


# for CLI
def main():
    """API for commad line
    """
    args = args_cfg()
    if args.install:
        install(args)
    else:
        print_info()
        return pipeline_predict(args.pipeline, args.model, args.input,
                                args.output, args.device)
