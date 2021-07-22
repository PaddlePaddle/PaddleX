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
from paddlex.seg import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='HumanSeg training')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='Model path for quant',
        type=str,
        default='output/best_model')
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size',
        type=int,
        default=1)
    parser.add_argument(
        '--batch_nums',
        dest='batch_nums',
        help='Batch number for quant',
        type=int,
        default=10)
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='the root directory of dataset',
        type=str)
    parser.add_argument(
        '--quant_list',
        dest='quant_list',
        help='Image file list for model quantization, it can be vat.txt or train.txt',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the quant model',
        type=str,
        default='./output/quant_offline')
    parser.add_argument(
        "--image_shape",
        dest="image_shape",
        help="The image shape for net inputs.",
        nargs=2,
        default=[192, 192],
        type=int)
    return parser.parse_args()


def evaluate(args):
    eval_transforms = transforms.Compose(
        [transforms.Resize(args.image_shape), transforms.Normalize()])

    eval_dataset = pdx.datasets.SegDataset(
        data_dir=args.data_dir,
        file_list=args.quant_list,
        transforms=eval_transforms)

    model = pdx.load_model(args.model_dir)
    pdx.slim.export_quant_model(model, eval_dataset, args.batch_size,
                                args.batch_nums, args.save_dir)


if __name__ == '__main__':
    args = parse_args()

    evaluate(args)
