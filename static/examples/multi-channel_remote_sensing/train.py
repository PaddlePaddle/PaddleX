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

import os.path as osp
import argparse
from paddlex.seg import transforms
import paddlex as pdx


def parse_args():
    parser = argparse.ArgumentParser(description='RemoteSensing training')
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='dataset directory',
        default=None,
        type=str)
    parser.add_argument(
        '--train_file_list',
        dest='train_file_list',
        help='train file_list',
        default=None,
        type=str)
    parser.add_argument(
        '--eval_file_list',
        dest='eval_file_list',
        help='eval file_list',
        default=None,
        type=str)
    parser.add_argument(
        '--label_list',
        dest='label_list',
        help='label_list file',
        default=None,
        type=str)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='model save directory',
        default=None,
        type=str)
    parser.add_argument(
        '--num_classes',
        dest='num_classes',
        help='Number of classes',
        default=None,
        type=int)
    parser.add_argument(
        '--channel',
        dest='channel',
        help='number of data channel',
        default=3,
        type=int)
    parser.add_argument(
        '--clip_min_value',
        dest='clip_min_value',
        help='Min values for clipping data',
        nargs='+',
        default=None,
        type=int)
    parser.add_argument(
        '--clip_max_value',
        dest='clip_max_value',
        help='Max values for clipping data',
        nargs='+',
        default=None,
        type=int)
    parser.add_argument(
        '--mean',
        dest='mean',
        help='Data means',
        nargs='+',
        default=None,
        type=float)
    parser.add_argument(
        '--std',
        dest='std',
        help='Data standard deviation',
        nargs='+',
        default=None,
        type=float)
    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        help='number of traing epochs',
        default=100,
        type=int)
    parser.add_argument(
        '--train_batch_size',
        dest='train_batch_size',
        help='training batch size',
        default=4,
        type=int)
    parser.add_argument(
        '--lr', dest='lr', help='learning rate', default=0.01, type=float)
    return parser.parse_args()


args = parse_args()
data_dir = args.data_dir
train_list = args.train_file_list
val_list = args.eval_file_list
label_list = args.label_list
save_dir = args.save_dir
num_classes = args.num_classes
channel = args.channel
clip_min_value = args.clip_min_value
clip_max_value = args.clip_max_value
mean = args.mean
std = args.std
num_epochs = args.num_epochs
train_batch_size = args.train_batch_size
lr = args.lr

# 定义训练和验证时的transforms
train_transforms = transforms.Compose([
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ResizeStepScaling(0.5, 2.0, 0.25),
    transforms.RandomPaddingCrop(im_padding_value=[1000] * channel),
    transforms.Clip(
        min_val=clip_min_value, max_val=clip_max_value),
    transforms.Normalize(
        min_val=clip_min_value, max_val=clip_max_value, mean=mean, std=std),
])

eval_transforms = transforms.Compose([
    transforms.Clip(
        min_val=clip_min_value, max_val=clip_max_value),
    transforms.Normalize(
        min_val=clip_min_value, max_val=clip_max_value, mean=mean, std=std),
])

train_dataset = pdx.datasets.SegDataset(
    data_dir=data_dir,
    file_list=train_list,
    label_list=label_list,
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.SegDataset(
    data_dir=data_dir,
    file_list=val_list,
    label_list=label_list,
    transforms=eval_transforms)

model = pdx.seg.UNet(num_classes=num_classes, input_channel=channel)

model.train(
    num_epochs=num_epochs,
    train_dataset=train_dataset,
    train_batch_size=train_batch_size,
    eval_dataset=eval_dataset,
    save_interval_epochs=5,
    log_interval_steps=10,
    save_dir=save_dir,
    learning_rate=lr,
    use_vdl=True)
