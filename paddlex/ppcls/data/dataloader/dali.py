# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division

import copy
import os

import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import paddle
from nvidia.dali import fn
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.paddle import DALIGenericIterator


class HybridTrainPipe(Pipeline):
    def __init__(self,
                 file_root,
                 file_list,
                 batch_size,
                 resize_shorter,
                 crop,
                 min_area,
                 lower,
                 upper,
                 interp,
                 mean,
                 std,
                 device_id,
                 shard_id=0,
                 num_shards=1,
                 random_shuffle=True,
                 num_threads=4,
                 seed=42,
                 pad_output=False,
                 output_dtype=types.FLOAT,
                 dataset='Train'):
        super(HybridTrainPipe, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.input = ops.readers.File(
            file_root=file_root,
            file_list=file_list,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=random_shuffle)
        # set internal nvJPEG buffers size to handle full-sized ImageNet images
        # without additional reallocations
        device_memory_padding = 211025920
        host_memory_padding = 140544512
        self.decode = ops.decoders.ImageRandomCrop(
            device='mixed',
            output_type=types.DALIImageType.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            random_aspect_ratio=[lower, upper],
            random_area=[min_area, 1.0],
            num_attempts=100)
        self.res = ops.Resize(
            device='gpu', resize_x=crop, resize_y=crop, interp_type=interp)
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            dtype=output_dtype,
            output_layout='CHW',
            crop=(crop, crop),
            mean=mean,
            std=std,
            pad_output=pad_output)
        self.coin = ops.random.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.DALIDataType.INT64, device="gpu")

    def define_graph(self):
        rng = self.coin()
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.to_int64(labels.gpu())]

    def __len__(self):
        return self.epoch_size("Reader")


class HybridValPipe(Pipeline):
    def __init__(self,
                 file_root,
                 file_list,
                 batch_size,
                 resize_shorter,
                 crop,
                 interp,
                 mean,
                 std,
                 device_id,
                 shard_id=0,
                 num_shards=1,
                 random_shuffle=False,
                 num_threads=4,
                 seed=42,
                 pad_output=False,
                 output_dtype=types.FLOAT):
        super(HybridValPipe, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.input = ops.readers.File(
            file_root=file_root,
            file_list=file_list,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=random_shuffle)
        self.decode = ops.decoders.Image(device="mixed")
        self.res = ops.Resize(
            device="gpu", resize_shorter=resize_shorter, interp_type=interp)
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            dtype=output_dtype,
            output_layout='CHW',
            crop=(crop, crop),
            mean=mean,
            std=std,
            pad_output=pad_output)
        self.to_int64 = ops.Cast(dtype=types.DALIDataType.INT64, device="gpu")

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.to_int64(labels.gpu())]

    def __len__(self):
        return self.epoch_size("Reader")


def dali_dataloader(config, mode, device, seed=None):
    assert "gpu" in device, "gpu training is required for DALI"
    device_id = int(device.split(':')[1])
    config_dataloader = config[mode]
    seed = 42 if seed is None else seed
    ops = [
        list(x.keys())[0]
        for x in config_dataloader["dataset"]["transform_ops"]
    ]
    support_ops_train = [
        "DecodeImage", "NormalizeImage", "RandFlipImage", "RandCropImage"
    ]
    support_ops_eval = [
        "DecodeImage", "ResizeImage", "CropImage", "NormalizeImage"
    ]

    if mode.lower() == 'train':
        assert set(ops) == set(
            support_ops_train
        ), "The supported trasform_ops for train_dataset in dali is : {}".format(
            ",".join(support_ops_train))
    else:
        assert set(ops) == set(
            support_ops_eval
        ), "The supported trasform_ops for eval_dataset in dali is : {}".format(
            ",".join(support_ops_eval))

    normalize_ops = [
        op for op in config_dataloader["dataset"]["transform_ops"]
        if "NormalizeImage" in op
    ][0]["NormalizeImage"]
    channel_num = normalize_ops.get("channel_num", 3)
    output_dtype = types.FLOAT16 if normalize_ops.get("output_fp16",
                                                      False) else types.FLOAT

    env = os.environ
    #  assert float(env.get('FLAGS_fraction_of_gpu_memory_to_use', 0.92)) < 0.9, \
    #      "Please leave enough GPU memory for DALI workspace, e.g., by setting" \
    #      " `export FLAGS_fraction_of_gpu_memory_to_use=0.8`"

    gpu_num = paddle.distributed.get_world_size()

    batch_size = config_dataloader["sampler"]["batch_size"]

    file_root = config_dataloader["dataset"]["image_root"]
    file_list = config_dataloader["dataset"]["cls_label_path"]

    interp = 1  # settings.interpolation or 1  # default to linear
    interp_map = {
        0: types.DALIInterpType.INTERP_NN,  # cv2.INTER_NEAREST
        1: types.DALIInterpType.INTERP_LINEAR,  # cv2.INTER_LINEAR
        2: types.DALIInterpType.INTERP_CUBIC,  # cv2.INTER_CUBIC
        3: types.DALIInterpType.
        INTERP_LANCZOS3,  # XXX use LANCZOS3 for cv2.INTER_LANCZOS4
    }

    assert interp in interp_map, "interpolation method not supported by DALI"
    interp = interp_map[interp]
    pad_output = channel_num == 4

    transforms = {
        k: v
        for d in config_dataloader["dataset"]["transform_ops"]
        for k, v in d.items()
    }

    scale = transforms["NormalizeImage"].get("scale", 1.0 / 255)
    scale = eval(scale) if isinstance(scale, str) else scale
    mean = transforms["NormalizeImage"].get("mean", [0.485, 0.456, 0.406])
    std = transforms["NormalizeImage"].get("std", [0.229, 0.224, 0.225])
    mean = [v / scale for v in mean]
    std = [v / scale for v in std]

    sampler_name = config_dataloader["sampler"].get("name",
                                                    "DistributedBatchSampler")
    assert sampler_name in ["DistributedBatchSampler", "BatchSampler"]

    if mode.lower() == "train":
        resize_shorter = 256
        crop = transforms["RandCropImage"]["size"]
        scale = transforms["RandCropImage"].get("scale", [0.08, 1.])
        ratio = transforms["RandCropImage"].get("ratio", [3.0 / 4, 4.0 / 3])
        min_area = scale[0]
        lower = ratio[0]
        upper = ratio[1]

        if 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env:
            shard_id = int(env['PADDLE_TRAINER_ID'])
            num_shards = int(env['PADDLE_TRAINERS_NUM'])
            device_id = int(env['FLAGS_selected_gpus'])
            pipe = HybridTrainPipe(
                file_root,
                file_list,
                batch_size,
                resize_shorter,
                crop,
                min_area,
                lower,
                upper,
                interp,
                mean,
                std,
                device_id,
                shard_id,
                num_shards,
                seed=seed + shard_id,
                pad_output=pad_output,
                output_dtype=output_dtype)
            pipe.build()
            pipelines = [pipe]
            #  sample_per_shard = len(pipe) // num_shards
        else:
            pipe = HybridTrainPipe(
                file_root,
                file_list,
                batch_size,
                resize_shorter,
                crop,
                min_area,
                lower,
                upper,
                interp,
                mean,
                std,
                device_id=device_id,
                shard_id=0,
                num_shards=1,
                seed=seed,
                pad_output=pad_output,
                output_dtype=output_dtype)
            pipe.build()
            pipelines = [pipe]
            #  sample_per_shard = len(pipelines[0])
        return DALIGenericIterator(
            pipelines, ['data', 'label'], reader_name='Reader')
    else:
        resize_shorter = transforms["ResizeImage"].get("resize_short", 256)
        crop = transforms["CropImage"]["size"]
        if 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env and sampler_name == "DistributedBatchSampler":
            shard_id = int(env['PADDLE_TRAINER_ID'])
            num_shards = int(env['PADDLE_TRAINERS_NUM'])
            device_id = int(env['FLAGS_selected_gpus'])

            pipe = HybridValPipe(
                file_root,
                file_list,
                batch_size,
                resize_shorter,
                crop,
                interp,
                mean,
                std,
                device_id=device_id,
                shard_id=shard_id,
                num_shards=num_shards,
                pad_output=pad_output,
                output_dtype=output_dtype)
        else:
            pipe = HybridValPipe(
                file_root,
                file_list,
                batch_size,
                resize_shorter,
                crop,
                interp,
                mean,
                std,
                device_id=device_id,
                pad_output=pad_output,
                output_dtype=output_dtype)
        pipe.build()
        return DALIGenericIterator(
            [pipe], ['data', 'label'], reader_name="Reader")
