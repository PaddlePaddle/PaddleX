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

import six
import sys
from paddle.io import DataLoader


class BaseDataLoader(object):
    def __init__(self, dataset, batch_sampler, use_shared_memory):
        self._batch_transforms = dataset.batch_transforms
        self._batch_sampler = batch_sampler
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self._batch_transforms,
            num_workers=self.dataset.num_workers,
            return_list=True,
            use_shared_memory=use_shared_memory)
        self.loader = iter(self.dataloader)

    def __call__(self):
        return self

    def __len__(self):
        return len(self._batch_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration:
            self.loader = iter(self.dataloader)
            six.reraise(*sys.exc_info())

    def next(self):
        # python2 compatibility
        return self.__next__()
