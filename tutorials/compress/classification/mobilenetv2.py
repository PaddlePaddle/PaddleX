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

import argparse
import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.cls import transforms
import paddlex as pdx


def train(model_dir=None, sensitivities_file=None, eval_metric_loss=0.05):
    # 下载和解压蔬菜分类数据集
    veg_dataset = 'https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz'
    pdx.utils.download_and_decompress(veg_dataset, path='./')

    # 定义训练和验证时的transforms
    train_transforms = transforms.Compose([
        transforms.RandomCrop(crop_size=224),
        transforms.RandomHorizontalFlip(), transforms.Normalize()
    ])
    eval_transforms = transforms.Compose([
        transforms.ResizeByShort(short_size=256),
        transforms.CenterCrop(crop_size=224), transforms.Normalize()
    ])

    # 定义训练和验证所用的数据集
    train_dataset = pdx.datasets.ImageNet(
        data_dir='vegetables_cls',
        file_list='vegetables_cls/train_list.txt',
        label_list='vegetables_cls/labels.txt',
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.ImageNet(
        data_dir='vegetables_cls',
        file_list='vegetables_cls/val_list.txt',
        label_list='vegetables_cls/labels.txt',
        transforms=eval_transforms)

    num_classes = len(train_dataset.labels)
    model = pdx.cls.MobileNetV2(num_classes=num_classes)

    if model_dir is None:
        # 使用imagenet数据集预训练模型权重
        pretrain_weights = "IMAGENET"
    else:
        # 使用传入的model_dir作为预训练模型权重
        assert os.path.isdir(model_dir), "Path {} is not a directory".format(
            model_dir)
        pretrain_weights = model_dir

    save_dir = './output/mobilenetv2'
    if sensitivities_file is not None:
        # DEFAULT 指使用模型预置的参数敏感度信息作为裁剪依据
        if sensitivities_file != "DEFAULT":
            assert os.path.exists(
                sensitivities_file), "Path {} not exist".format(
                    sensitivities_file)
        save_dir = './output/mobilenetv2_prune'

    model.train(
        num_epochs=10,
        train_dataset=train_dataset,
        train_batch_size=32,
        eval_dataset=eval_dataset,
        lr_decay_epochs=[4, 6, 8],
        learning_rate=0.025,
        pretrain_weights=pretrain_weights,
        save_dir=save_dir,
        use_vdl=True,
        sensitivities_file=sensitivities_file,
        eval_metric_loss=eval_metric_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir", default=None, type=str, help="The model path.")
    parser.add_argument(
        "--sensitivities_file",
        default=None,
        type=str,
        help="The sensitivities file path.")
    parser.add_argument(
        "--eval_metric_loss",
        default=0.05,
        type=float,
        help="The loss threshold.")

    args = parser.parse_args()
    train(args.model_dir, args.sensitivities_file, args.eval_metric_loss)
