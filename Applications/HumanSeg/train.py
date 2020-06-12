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

import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 使用CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse

import paddlex as pdx
from paddlex.seg import transforms

MODEL_TYPE = ['HumanSegMobile', 'HumanSegServer']


def parse_args():
    parser = argparse.ArgumentParser(description='HumanSeg training')
    parser.add_argument(
        '--model_type',
        dest='model_type',
        help="Model type for traing, which is one of ('HumanSegMobile', 'HumanSegServer')",
        type=str,
        default='HumanSegMobile')
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='The root directory of dataset',
        type=str)
    parser.add_argument(
        '--train_list',
        dest='train_list',
        help='Train list file of dataset',
        type=str)
    parser.add_argument(
        '--val_list',
        dest='val_list',
        help='Val list file of dataset',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--num_classes',
        dest='num_classes',
        help='Number of classes',
        type=int,
        default=2)
    parser.add_argument(
        "--image_shape",
        dest="image_shape",
        help="The image shape for net inputs.",
        nargs=2,
        default=[192, 192],
        type=int)
    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        help='Number epochs for training',
        type=int,
        default=100)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size',
        type=int,
        default=128)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=0.01)
    parser.add_argument(
        '--pretrain_weights',
        dest='pretrain_weights',
        help='The path of pretrianed weight',
        type=str,
        default=None)
    parser.add_argument(
        '--resume_checkpoint',
        dest='resume_checkpoint',
        help='The path of resume checkpoint',
        type=str,
        default=None)
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to use visualdl',
        action='store_true')
    parser.add_argument(
        '--save_interval_epochs',
        dest='save_interval_epochs',
        help='The interval epochs for save a model snapshot',
        type=int,
        default=5)

    return parser.parse_args()


def train(args):
    train_transforms = transforms.Compose([
        transforms.Resize(args.image_shape), transforms.RandomHorizontalFlip(),
        transforms.Normalize()
    ])

    eval_transforms = transforms.Compose(
        [transforms.Resize(args.image_shape), transforms.Normalize()])

    train_dataset = pdx.datasets.SegDataset(
        data_dir=args.data_dir,
        file_list=args.train_list,
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.SegDataset(
        data_dir=args.data_dir,
        file_list=args.val_list,
        transforms=eval_transforms)

    if args.model_type == 'HumanSegMobile':
        model = pdx.seg.HRNet(
            num_classes=args.num_classes, width='18_small_v1')
    elif args.model_type == 'HumanSegServer':
        model = pdx.seg.DeepLabv3p(
            num_classes=args.num_classes, backbone='Xception65')
    else:
        raise ValueError(
            "--model_type: {} is set wrong, it shold be one of ('HumanSegMobile', "
            "'HumanSegLite', 'HumanSegServer')".format(args.model_type))
    model.train(
        num_epochs=args.num_epochs,
        train_dataset=train_dataset,
        train_batch_size=args.batch_size,
        eval_dataset=eval_dataset,
        save_interval_epochs=args.save_interval_epochs,
        learning_rate=args.learning_rate,
        pretrain_weights=args.pretrain_weights,
        resume_checkpoint=args.resume_checkpoint,
        save_dir=args.save_dir,
        use_vdl=args.use_vdl)


if __name__ == '__main__':
    args = parse_args()
    train(args)
