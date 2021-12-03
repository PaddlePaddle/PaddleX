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

from __future__ import absolute_import
from __future__ import division

from paddle.io import DistributedBatchSampler, Sampler

from paddlex.ppcls.utils import logger
from paddlex.ppcls.data.dataloader.mix_dataset import MixDataset
from paddlex.ppcls.data import dataloader


class MixSampler(DistributedBatchSampler):
    def __init__(self, dataset, batch_size, sample_configs, iter_per_epoch):
        super().__init__(dataset, batch_size)
        assert isinstance(dataset,
                          MixDataset), "MixSampler only support MixDataset"
        self.sampler_list = []
        self.batch_size = batch_size
        self.start_list = []
        self.length = iter_per_epoch
        dataset_list = dataset.get_dataset_list()
        batch_size_left = self.batch_size
        self.iter_list = []
        for i, config_i in enumerate(sample_configs):
            self.start_list.append(dataset_list[i][1])
            sample_method = config_i.pop("name")
            ratio_i = config_i.pop("ratio")
            if i < len(sample_configs) - 1:
                batch_size_i = int(self.batch_size * ratio_i)
                batch_size_left -= batch_size_i
            else:
                batch_size_i = batch_size_left
            assert batch_size_i <= len(dataset_list[i][2])
            config_i["batch_size"] = batch_size_i
            if sample_method == "DistributedBatchSampler":
                sampler_i = DistributedBatchSampler(dataset_list[i][2],
                                                    **config_i)
            else:
                sampler_i = getattr(dataloader, sample_method)(
                    dataset_list[i][2], **config_i)
            self.sampler_list.append(sampler_i)
            self.iter_list.append(iter(sampler_i))
            self.length += len(dataset_list[i][2]) * ratio_i
            self.iter_counter = 0

    def __iter__(self):
        while self.iter_counter < self.length:
            batch = []
            for i, iter_i in enumerate(self.iter_list):
                batch_i = next(iter_i, None)
                if batch_i is None:
                    iter_i = iter(self.sampler_list[i])
                    self.iter_list[i] = iter_i
                    batch_i = next(iter_i, None)
                    assert batch_i is not None, "dataset {} return None".format(
                        i)
                batch += [idx + self.start_list[i] for idx in batch_i]
            if len(batch) == self.batch_size:
                self.iter_counter += 1
                yield batch
            else:
                logger.info("Some dataset reaches end")
        self.iter_counter = 0

    def __len__(self):
        return self.length
