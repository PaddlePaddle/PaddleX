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

import json
import time
import uuid

from paddlex import create_pipeline
from paddlex.inference import load_pipeline_config
from paddlex.utils.device import constr_device
from pydantic import ValidationError

from . import constants, env, logging, protocol
from .config import create_app_config
from .lazy_mods import pb_utils


class BaseTritonPythonModel(object):
    @property
    def id(self):
        # TODO: Add model name
        return self._id

    @property
    def pipeline_creation_kwargs(self):
        return {}

    def initialize(self, args):
        self.triton_model_config = json.loads(args["model_config"])
        logging.info("Triton model config: %s", self.triton_model_config)

        self.input_names = []
        for input_config in self.triton_model_config["input"]:
            self.input_names.append(input_config["name"])
        if self.input_names != [constants.INPUT_NAME]:
            raise pb_utils.TritonModelException(
                f"Expected one and only one input named {repr(constants.INPUT_NAME)}, but got input names: {self.input_names}"
            )
        logging.info("Input names: %s", self.input_names)

        self.output_names = []
        self.output_dtypes = []
        for output_config in self.triton_model_config["output"]:
            self.output_names.append(output_config["name"])
            self.output_dtypes.append(output_config["data_type"])
        if self.output_names != [constants.OUTPUT_NAME]:
            raise pb_utils.TritonModelException(
                f"Expected one and only one output named {repr(constants.OUTPUT_NAME)}, but got output names: {self.output_names}"
            )
        logging.info("Output names: %s", self.output_names)

        if args["model_instance_kind"] == "GPU":
            self._device_type = "gpu"
            self._device_id = int(args["model_instance_device_id"])
        elif args["model_instance_kind"] == "CPU":
            self._device_type = "cpu"
            self._device_id = None
        else:
            raise pb_utils.TritonModelException(
                f"Unsupported model instance kind: {args['model_instance_kind']}"
            )

        self.pipeline_config = load_pipeline_config(env.PIPELINE_CONFIG_PATH)
        use_hpip = env.USE_HPIP == "1"
        self.pipeline = self._create_pipeline(self.pipeline_config, use_hpip)
        self.app_config = create_app_config(self.pipeline_config)

        self._id = self._generate_model_id()
        logging.info("%s initialized successfully", self.id)

    def execute(self, requests):
        batch_id = self._generate_batch_id()
        tokens = logging.set_context_vars(self.id, batch_id)
        logging.info("Received batch of size %s", len(requests))
        start_time = time.perf_counter()

        try:
            inputs = {}
            outputs = {}
            log_ids = []
            for i, request in enumerate(requests):
                log_id = protocol.generate_log_id()
                logging.info("Request %s received", log_id)
                log_ids.append(log_id)
                input_ = pb_utils.get_input_tensor_by_name(
                    request, constants.INPUT_NAME
                )
                input_ = input_.as_numpy()
                input_model_type = self.get_input_model_type()
                try:
                    input_ = protocol.parse_triton_input(input_, input_model_type)
                    inputs[i] = input_
                except ValidationError as e:
                    output = protocol.create_aistudio_output_without_result(
                        422, str(e), log_id=log_id
                    )
                    outputs[i] = output

            if inputs:
                try:
                    result_or_output_lst = self.run_batch(
                        inputs.values(), [log_ids[i] for i in inputs.keys()], batch_id
                    )
                except Exception as e:
                    logging.error("Unhandled exception", exc_info=e)
                    for i in inputs.keys():
                        outputs[i] = protocol.create_aistudio_output_without_result(
                            500, "Internal server error", log_id=log_ids[i]
                        )
                else:
                    result_model_type = self.get_result_model_type()
                    for i, item in enumerate(result_or_output_lst):
                        if isinstance(item, result_model_type):
                            outputs[i] = protocol.create_aistudio_output_with_result(
                                item,
                                log_id=log_ids[i],
                            )
                        else:
                            outputs[i] = item

            assert len(outputs) == len(
                requests
            ), f"The number of outputs ({len(outputs)}) does not match the number of requests ({len(requests)})"

            responses = []
            for i in range(len(requests)):
                output = outputs[i]
                output = protocol.create_triton_output(output)
                output = pb_utils.Tensor(constants.OUTPUT_NAME, output)
                response = pb_utils.InferenceResponse(output_tensors=[output])
                responses.append(response)
        except Exception as e:
            logging.error("Unhandled exception", exc_info=e)
            responses = [
                pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError("An error occurred during execution"),
                )
                for _ in requests
            ]
        finally:
            end_time = time.perf_counter()
            logging.info("Time taken: %.3f ms", (end_time - start_time) * 1000)
            logging.reset_context_vars(*tokens)

        return responses

    def get_input_model_type(self):
        raise NotImplementedError

    def get_result_model_type(self):
        raise NotImplementedError

    def run(self, input, log_id):
        raise NotImplementedError

    def run_batch(self, inputs, log_ids, batch_id):
        if len(inputs) != len(log_ids):
            raise ValueError(
                "The number of `inputs` does not match the number of `log_ids`"
            )
        outputs = []
        for inp, log_id in zip(inputs, log_ids):
            out = self.run(inp, log_id)
            outputs.append(out)
        return outputs

    def _create_pipeline(self, config, use_hpip):
        if self._device_id is not None:
            device = constr_device(self._device_type, [self._device_id])
        else:
            device = self._device_type
        pipeline = create_pipeline(
            config=config,
            device=device,
            use_hpip=use_hpip,
            **self.pipeline_creation_kwargs,
        )
        return pipeline

    def _generate_model_id(self):
        return uuid.uuid4().hex

    def _generate_batch_id(self):
        return uuid.uuid4().hex
