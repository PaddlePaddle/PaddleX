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

import sys
import os
import argparse
import deploy


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        "-m",
        type=str,
        default=None,
        help="path to openvino model .xml file")
    parser.add_argument(
        "--img", "-i", type=str, default=None, help="path to an image files")

    parser.add_argument(
        "--img_list", "-l", type=str, default=None, help="Path to a imglist")

    parser.add_argument(
        "--cfg_file",
        "-c",
        type=str,
        default=None,
        help="Path to PaddelX model yml file")

    parser.add_argument(
        "--thread_num",
        "-t",
        type=int,
        default=1,
        help="Path to PaddelX model yml file")

    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()
    model_nb = args.model_dir
    model_yaml = args.cfg_file
    thread_num = args.thread_num
    #model init
    predictor = deploy.Predictor(model_nb, model_yaml, thread_num)

    #predict
    if (args.img_list != None):
        f = open(args.img_list)
        lines = f.readlines()
        for im_path in lines:
            print(im_path)
            predictor.predict(im_path.strip('\n'))
        f.close()
    else:
        im_path = args.img
        predictor.predict(im_path)


if __name__ == "__main__":
    main()
