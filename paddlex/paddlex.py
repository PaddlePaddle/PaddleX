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

from .utils.config import AttrDict
from .modules.base.predictor import build_predictor, BasePredictor
from .repo_manager import setup, get_all_supported_repo_names


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

    ################# infer #################
    parser.add_argument('--predict', action='store_true', default=True, help="")
    parser.add_argument('--model_name', type=str, help="")
    parser.add_argument('--model', type=str, help="")
    parser.add_argument('--input_path', type=str, help="")
    parser.add_argument('--output', type=str, help="")
    parser.add_argument('--device', type=str, default='gpu:0', help="")

    return parser.parse_args()


def get_all_models():
    """Get all models that have been registered
    """
    all_models = BasePredictor.all()
    model_map = {}
    for model in all_models:
        module = all_models[model].__name__
        if module not in model_map:
            model_map[module] = []
        model_map[module].append(model)
    return model_map


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

    models_table = PrettyTable()
    models_table.field_names = ["Modules", "Models"]
    model_map = get_all_models()
    for module in model_map:
        models = model_map[module]
        models_table.add_row(
            [
                textwrap.fill(
                    f"{module}", width=total_width // 5), textwrap.fill(
                        "  ".join(models), width=total_width * 4 // 5)
            ],
            divider=True)

    table_width = len(str(models_table).split("\n")[0])

    print("{}".format("-" * table_width))
    print("PaddleX".center(table_width))
    print(models_table)
    print("Powered by PaddlePaddle!".rjust(table_width))
    print("{}".format("-" * table_width))


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


def build_predict_config(model_name, model, input_path, device, output):
    """build predict config for paddlex
    """

    def dict2attrdict(dict_obj):
        """convert dict object to AttrDict
        """
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                dict_obj[key] = dict2attrdict(value)
        return AttrDict(dict_obj)

    config = {
        'Global': {
            'model': model_name,
            'device': device,
            'output': output
        },
        'Predict': {
            'model_dir': model,
            'input_path': input_path,
        }
    }

    return dict2attrdict(config)


def predict(model_name, model, input_path, device, output):
    """predict using paddlex
    """
    config = build_predict_config(model_name, model, input_path, device, output)
    predict = build_predictor(config)
    return predict()


# for CLI
def main():
    """API for commad line
    """
    args = args_cfg()
    if args.install:
        install(args)
    else:
        print_info()
        return predict(args.model_name, args.model, args.input_path,
                       args.device, args.output)
