# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import paddlex as pdx
import paddlex.utils.logging as logging
from paddlex.seg import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='HumanSeg training')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='Model path for evaluating',
        type=str,
        default='output/best_model')
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='The root directory of dataset',
        type=str)
    parser.add_argument(
        '--val_list',
        dest='val_list',
        help='Val list file of dataset',
        type=str,
        default=None)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size',
        type=int,
        default=128)
    parser.add_argument(
        "--image_shape",
        dest="image_shape",
        help="The image shape for net inputs.",
        nargs=2,
        default=[192, 192],
        type=int)
    return parser.parse_args()


def dict2str(dict_input):
    out = ''
    for k, v in dict_input.items():
        try:
            v = round(float(v), 6)
        except:
            pass
        out = out + '{}={}, '.format(k, v)
    return out.strip(', ')


def evaluate(args):
    eval_transforms = transforms.Compose(
        [transforms.Resize(args.image_shape), transforms.Normalize()])

    eval_dataset = pdx.datasets.SegDataset(
        data_dir=args.data_dir,
        file_list=args.val_list,
        transforms=eval_transforms)

    model = pdx.load_model(args.model_dir)
    metrics = model.evaluate(eval_dataset, args.batch_size)
    logging.info('[EVAL] Finished, {} .'.format(dict2str(metrics)))


if __name__ == '__main__':
    args = parse_args()

    evaluate(args)
