# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from abc import ABC, abstractmethod
from ... import c_lib_wrap as C


class ProcessorManager:
    def __init__(self):
        self._manager = None

    def run(self, input_ims):
        """Process input image

        :param: input_ims: (list of numpy.ndarray) The input images
        :return: list of FDTensor
        """
        return self._manager.run(input_ims)

    def use_cuda(self, enable_cv_cuda=False, gpu_id=-1):
        """Use CUDA processors

        :param: enable_cv_cuda: Ture: use CV-CUDA, False: use CUDA only
        :param: gpu_id: GPU device id
        """
        return self._manager.use_cuda(enable_cv_cuda, gpu_id)


class PyProcessorManager(ABC):
    """
    PyProcessorManager is used to define a customized processor in python
    """

    def __init__(self):
        self._manager = C.vision.processors.ProcessorManager()

    def use_cuda(self, enable_cv_cuda=False, gpu_id=-1):
        """Use CUDA processors

        :param: enable_cv_cuda: Ture: use CV-CUDA, False: use CUDA only
        :param: gpu_id: GPU device id
        """
        return self._manager.use_cuda(enable_cv_cuda, gpu_id)

    def __call__(self, images):
        image_batch = C.vision.FDMatBatch()
        image_batch.from_mats(images)

        self._manager.pre_apply(image_batch)
        outputs = self.apply(image_batch)
        self._manager.post_apply()
        return outputs

    @abstractmethod
    def apply(self, image_batch):
        print("This function has to be implemented.")
        return []
