# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import inspect
from copy import deepcopy
from abc import ABC
from types import GeneratorType

from ...utils import logging


class BaseComponent(ABC):

    YIELD_BATCH = True
    KEEP_INPUT = True
    ENABLE_BATCH = False

    INPUT_KEYS = None
    OUTPUT_KEYS = None

    def __init__(self):
        self.inputs = self.DEAULT_INPUTS if hasattr(self, "DEAULT_INPUTS") else {}
        self.outputs = self.DEAULT_OUTPUTS if hasattr(self, "DEAULT_OUTPUTS") else {}

    def __call__(self, input_list):
        # use list type for batched data
        if not isinstance(input_list, list):
            input_list = [input_list]

        output_list = []
        for args, input_ in self._check_input(input_list):
            output = self.apply(**args)
            if not output:
                if self.YIELD_BATCH:
                    yield input_list
                else:
                    for item in input_list:
                        yield item

            # output may be a generator, when the apply() uses yield
            if isinstance(output, GeneratorType):
                # if output is a generator, use for-in to get every one batch output data and yield one by one
                for each_output in output:
                    reassemble_data = self._check_output(each_output, input_)
                    if self.YIELD_BATCH:
                        yield reassemble_data
                    else:
                        for item in reassemble_data:
                            yield item
            # if output is not a generator, process all data of that and yield, so use output_list to collect all reassemble_data
            else:
                reassemble_data = self._check_output(output, input_)
                output_list.extend(reassemble_data)

        # avoid yielding output_list when the output is a generator
        if len(output_list) > 0:
            if self.YIELD_BATCH:
                yield output_list
            else:
                for item in output_list:
                    yield item

    def _check_input(self, input_list):
        # check if the value of input data meets the requirements of apply(),
        # and reassemble the parameters of apply() from input_list
        def _check_type(input_):
            if not isinstance(input_, dict):
                if len(self.inputs) == 1:
                    key = list(self.inputs.keys())[0]
                    input_ = {key: input_}
                else:
                    raise Exception(
                        f"The input must be a dict or a list of dict, unless the input of the component only requires one argument, but the component({self.__class__.__name__}) requires {list(self.inputs.keys())}!"
                    )
            return input_

        def _check_args_key(args):
            sig = inspect.signature(self.apply)
            for param in sig.parameters.values():
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    logging.debug(
                        f"The apply function parameter of {self.__class__.__name__} is **kwargs, so would not inspect!"
                    )
                    continue
                if param.default == inspect.Parameter.empty and param.name not in args:
                    raise Exception(
                        f"The parameter ({param.name}) is needed by {self.__class__.__name__}, but {list(args.keys())} only found!"
                    )

        if self.need_batch_input:
            args = {}
            for input_ in input_list:
                input_ = _check_type(input_)
                for k, v in self.inputs.items():
                    if v not in input_:
                        raise Exception(
                            f"The value ({v}) is needed by {self.__class__.__name__}. But not found in Data ({input_.keys()})!"
                        )
                    if k not in args:
                        args[k] = []
                    args[k].append(input_.get(v))
                _check_args_key(args)
            reassemble_input = [(args, input_list)]
        else:
            reassemble_input = []
            for input_ in input_list:
                input_ = _check_type(input_)
                args = {}
                for k, v in self.inputs.items():
                    if v not in input_:
                        raise Exception(
                            f"The value ({v}) is needed by {self.__class__.__name__}. But not found in Data ({input_.keys()})!"
                        )
                    args[k] = input_.get(v)
                _check_args_key(args)
                reassemble_input.append((args, input_))
        return reassemble_input

    def _check_output(self, output, ori_data):
        # check if the value of apply() output data meets the requirements of setting
        # when the output data is list type, reassemble each of that
        if isinstance(output, list):
            if self.need_batch_input:
                assert isinstance(ori_data, list) and len(ori_data) == len(output)
                output_list = []
                for ori_item, output_item in zip(ori_data, output):
                    data = ori_item.copy() if self.keep_input else {}
                    if isinstance(self.outputs, type(None)):
                        logging.debug(
                            f"The `output_key` of {self.__class__.__name__} is None, so would not inspect!"
                        )
                        data.update(output_item)
                    else:
                        for k, v in self.outputs.items():
                            if k not in output_item:
                                raise Exception(
                                    f"The value ({k}) is needed by {self.__class__.__name__}. But not found in Data ({output_item.keys()})!"
                                )
                            data.update({v: output_item[k]})
                    output_list.append(data)
                return output_list
            else:
                assert isinstance(ori_data, dict)
                output_list = []
                for output_item in output:
                    data = ori_data.copy() if self.keep_input else {}
                    if isinstance(self.outputs, type(None)):
                        logging.debug(
                            f"The `output_key` of {self.__class__.__name__} is None, so would not inspect!"
                        )
                        data.update(output_item)
                    else:
                        for k, v in self.outputs.items():
                            if k not in output_item:
                                raise Exception(
                                    f"The value ({k}) is needed by {self.__class__.__name__}. But not found in Data ({output_item.keys()})!"
                                )
                            data.update({v: output_item[k]})
                    output_list.append(data)
                return output_list
        else:
            assert isinstance(ori_data, dict) and isinstance(output, dict)
            data = ori_data.copy() if self.keep_input else {}
            if isinstance(self.outputs, type(None)):
                logging.debug(
                    f"The `output_key` of {self.__class__.__name__} is None, so would not inspect!"
                )
                data.update(output)
            else:
                for k, v in self.outputs.items():
                    if k not in output:
                        raise Exception(
                            f"The value of key ({k}) is needed add to Data. But not found in output of {self.__class__.__name__}: ({output.keys()})!"
                        )
                    data.update({v: output[k]})
        return [data]

    def set_inputs(self, inputs):
        assert isinstance(inputs, dict)
        input_keys = deepcopy(self.INPUT_KEYS)

        # e.g, input_keys is None or []
        if input_keys is None or (
            isinstance(input_keys, list) and len(input_keys) == 0
        ):
            self.inputs = {}
            if inputs:
                raise Exception
            return

        # e.g, input_keys is 'img'
        if not isinstance(input_keys, list):
            input_keys = [[input_keys]]
        # e.g, input_keys is ['img'] or [['img']]
        elif len(input_keys) > 0:
            # e.g, input_keys is ['img']
            if not isinstance(input_keys[0], list):
                input_keys = [input_keys]

        ck_pass = False
        for key_group in input_keys:
            for key in key_group:
                if key not in inputs:
                    break
            # check pass
            else:
                ck_pass = True
            if ck_pass == True:
                break
        else:
            raise Exception(
                f"The input {input_keys} are needed by {self.__class__.__name__}. But only get: {list(inputs.keys())}"
            )
        self.inputs = inputs

    def set_outputs(self, outputs):
        assert isinstance(outputs, dict)
        output_keys = deepcopy(self.OUTPUT_KEYS)
        if not isinstance(output_keys, list):
            output_keys = [output_keys]

        for k in output_keys:
            if k not in outputs:
                logging.debug(
                    f"The output ({k}) of {self.__class__.__name__} would be abandon!"
                )
        self.outputs = outputs

    @classmethod
    def get_input_keys(cls) -> list:
        return cls.input_keys

    @classmethod
    def get_output_keys(cls) -> list:
        return cls.output_keys

    @property
    def need_batch_input(self):
        return getattr(self, "ENABLE_BATCH", False)

    @property
    def keep_input(self):
        return getattr(self, "KEEP_INPUT", True)

    @property
    def name(self):
        return getattr(self, "NAME", self.__class__.__name__)


class ComponentsEngine(object):
    def __init__(self, ops):
        self.ops = ops
        self.keys = list(ops.keys())

    def __call__(self, data, i=0):
        data_gen = self.ops[self.keys[i]](data)
        if i + 1 < len(self.ops):
            for data in data_gen:
                yield from self.__call__(data, i + 1)
        else:
            yield from data_gen
