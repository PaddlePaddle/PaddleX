#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import argparse
import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx


def train(model_dir, sensitivities_file, eval_metric_loss):
    # 下载和解压昆虫检测数据集
    insect_dataset = 'https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz'
    pdx.utils.download_and_decompress(insect_dataset, path='./')

    # 定义训练和验证时的transforms
    train_transforms = transforms.Compose([
        transforms.MixupImage(mixup_epoch=250), transforms.RandomDistort(),
        transforms.RandomExpand(), transforms.RandomCrop(), transforms.Resize(
            target_size=608, interp='RANDOM'),
        transforms.RandomHorizontalFlip(), transforms.Normalize()
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize(
            target_size=608, interp='CUBIC'), transforms.Normalize()
    ])

    # 定义训练和验证所用的数据集
    train_dataset = pdx.datasets.VOCDetection(
        data_dir='insect_det',
        file_list='insect_det/train_list.txt',
        label_list='insect_det/labels.txt',
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.VOCDetection(
        data_dir='insect_det',
        file_list='insect_det/val_list.txt',
        label_list='insect_det/labels.txt',
        transforms=eval_transforms)

    if model_dir is None:
        # 使用imagenet数据集上的预训练权重
        pretrain_weights = "IMAGENET"
    else:
        assert os.path.isdir(model_dir), "Path {} is not a directory".format(
            model_dir)
        pretrain_weights = model_dir
    save_dir = "output/yolov3_mobile"
    if sensitivities_file is not None:
        if sensitivities_file != 'DEFAULT':
            assert os.path.exists(
                sensitivities_file), "Path {} not exist".format(
                    sensitivities_file)
        save_dir = "output/yolov3_mobile_prune"

    num_classes = len(train_dataset.labels)
    model = pdx.det.YOLOv3(num_classes=num_classes)
    model.train(
        num_epochs=270,
        train_dataset=train_dataset,
        train_batch_size=8,
        eval_dataset=eval_dataset,
        learning_rate=0.000125,
        lr_decay_epochs=[210, 240],
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
