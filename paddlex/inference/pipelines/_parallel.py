# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import abc
from concurrent.futures import ThreadPoolExecutor

from ...utils import device as device_utils
from ..common.batch_sampler import ImageBatchSampler
from .base import BasePipeline


class MultiDeviceSimpleInferenceExecutor(object):
    def __init__(self, pipelines, batch_sampler, *, postprocess_result=None):
        super().__init__()
        self._pipelines = pipelines
        self._batch_sampler = batch_sampler
        self._postprocess_result = postprocess_result

    @property
    def pipelines(self):
        return self._pipelines

    def execute(
        self,
        input,
        *args,
        **kwargs,
    ):
        with ThreadPoolExecutor(max_workers=len(self._pipelines)) as pool:
            input_batches = self._batch_sampler(input)
            out_of_data = False
            while not out_of_data:
                input_future_pairs = []
                for pipeline in self._pipelines:
                    try:
                        input_batch = next(input_batches)
                    except StopIteration:
                        out_of_data = True
                        break
                    input_instances = input_batch.instances
                    future = pool.submit(
                        lambda pipeline, input_instances, args, kwargs: list(
                            pipeline.predict(input_instances, *args, **kwargs)
                        ),
                        pipeline,
                        input_instances,
                        args,
                        kwargs,
                    )
                    input_future_pairs.append((input_batch, future))

                # We synchronize here to keep things simple (no data
                # prefetching, no queues, no dedicated workers), although
                # it's less efficient.
                for input_batch, future in input_future_pairs:
                    result = future.result()
                    for input_path, result_item in zip(input_batch.input_paths, result):
                        result_item["input_path"] = input_path
                    if self._postprocess_result:
                        result = self._postprocess_result(result, input_batch)
                    yield from result


class AutoParallelSimpleInferencePipeline(BasePipeline):
    def __init__(
        self,
        config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._multi_device_inference = False
        if self.device is not None:
            device_type, device_ids = device_utils.parse_device(self.device)
            if device_ids is not None and len(device_ids) > 1:
                self._multi_device_inference = True
                self._pipelines = []
                for device_id in device_ids:
                    pipeline = self._create_internal_pipeline(
                        config, device_utils.constr_device(device_type, [device_id])
                    )
                    self._pipelines.append(pipeline)
                batch_size = self._get_batch_size(config)
                batch_sampler = self._create_batch_sampler(batch_size)
                self._executor = MultiDeviceSimpleInferenceExecutor(
                    self._pipelines,
                    batch_sampler,
                    postprocess_result=self._postprocess_result,
                )
        if not self._multi_device_inference:
            self._pipeline = self._create_internal_pipeline(config, self.device)

    @property
    def multi_device_inference(self):
        return self._multi_device_inference

    def __getattr__(self, name):
        if self._multi_device_inference:
            first_pipeline = self._executor.pipelines[0]
            return getattr(first_pipeline, name)
        else:
            return getattr(self._pipeline, name)

    def predict(
        self,
        input,
        *args,
        **kwargs,
    ):
        if self._multi_device_inference:
            yield from self._executor.execute(
                input,
                *args,
                **kwargs,
            )
        else:
            yield from self._pipeline.predict(
                input,
                *args,
                **kwargs,
            )

    @abc.abstractmethod
    def _create_internal_pipeline(self, config, device):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_batch_size(self, config):
        raise NotImplementedError

    @abc.abstractmethod
    def _create_batch_sampler(self, batch_size):
        raise NotImplementedError

    def _postprocess_result(self, result, input_batch):
        return result


class AutoParallelImageSimpleInferencePipeline(AutoParallelSimpleInferencePipeline):
    @property
    @abc.abstractmethod
    def _pipeline_cls(self):
        raise NotImplementedError

    def _create_internal_pipeline(self, config, device):
        return self._pipeline_cls(
            config,
            device=device,
            pp_option=self.pp_option,
            use_hpip=self.use_hpip,
            hpi_config=self.hpi_config,
        )

    def _create_batch_sampler(self, batch_size):
        return ImageBatchSampler(batch_size)

    def _postprocess_result(self, result, input_batch):
        for page_index, item in zip(input_batch.page_indexes, result):
            item["page_index"] = page_index
        return result
