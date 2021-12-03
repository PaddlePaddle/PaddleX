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
import numpy as np
import cv2
import utils
import argparse
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../..')))

import paddle
from paddle.distributed import ParallelEnv

from resnet import ResNet50
from paddlex.ppcls.utils.save_load import load_dygraph_pretrain


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_file", required=True, type=str)
    parser.add_argument("-c", "--channel_num", type=int)
    parser.add_argument("-p", "--pretrained_model", type=str)
    parser.add_argument("--show", type=str2bool, default=False)
    parser.add_argument("--interpolation", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--use_gpu", type=str2bool, default=True)

    return parser.parse_args()


def create_operators(interpolation=1):
    size = 224
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_scale = 1.0 / 255.0

    resize_op = utils.ResizeImage(
        resize_short=256, interpolation=interpolation)
    crop_op = utils.CropImage(size=(size, size))
    normalize_op = utils.NormalizeImage(
        scale=img_scale, mean=img_mean, std=img_std)
    totensor_op = utils.ToTensor()

    return [resize_op, crop_op, normalize_op, totensor_op]


def preprocess(data, ops):
    for op in ops:
        data = op(data)
    return data


def main():
    args = parse_args()
    operators = create_operators(args.interpolation)
    # assign the place
    place = 'gpu:{}'.format(ParallelEnv().dev_id) if args.use_gpu else 'cpu'
    place = paddle.set_device(place)

    net = ResNet50()
    load_dygraph_pretrain(net, args.pretrained_model)

    img = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
    data = preprocess(img, operators)
    data = np.expand_dims(data, axis=0)
    data = paddle.to_tensor(data)
    net.eval()
    _, fm = net(data)
    assert args.channel_num >= 0 and args.channel_num <= fm.shape[
        1], "the channel is out of the range, should be in {} but got {}".format(
            [0, fm.shape[1]], args.channel_num)

    fm = (np.squeeze(fm[0][args.channel_num].numpy()) * 255).astype(np.uint8)
    fm = cv2.resize(fm, (img.shape[1], img.shape[0]))
    if args.save_path is not None:
        print("the feature map is saved in path: {}".format(args.save_path))
        cv2.imwrite(args.save_path, fm)


if __name__ == "__main__":
    main()
