# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.contrib.slim.quantization.quantization_pass import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization.quantization_pass import AddQuantDequantPass
from paddle.fluid.contrib.slim.quantization.quantization_pass import _out_scale_op_list
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
import paddlex.utils.logging as logging
import paddle.fluid as fluid
import os
import re
import numpy as np
import time


def _load_variable_data(scope, var_name):
    '''
    Load variable value from scope
    '''
    var_node = scope.find_var(var_name)
    assert var_node is not None, \
        "Cannot find " + var_name + " in scope."
    return np.array(var_node.get_tensor())


class PaddleXPostTrainingQuantization(PostTrainingQuantization):
    def __init__(self,
                 executor,
                 dataset,
                 program,
                 inputs,
                 outputs,
                 batch_size=10,
                 batch_nums=None,
                 scope=None,
                 algo="KL",
                 quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
                 is_full_quantize=False,
                 is_use_cache_file=False,
                 cache_dir="./temp_post_training"):
        '''
        The class utilizes post training quantization methon to quantize the
        fp32 model. It uses calibrate data to calculate the scale factor of
        quantized variables, and inserts fake quant/dequant op to obtain the
        quantized model.
        Args:
            executor(fluid.Executor): The executor to load, run and save the
                quantized model.
            dataset(Python Iterator): The data Reader.
            program(fluid.Program): The paddle program, save the parameters for model.
            inputs(dict): The input of prigram.
            outputs(dict): The output of program.
            batch_size(int, optional): The batch size of DataLoader. Default is 10.
            batch_nums(int, optional): If batch_nums is not None, the number of
                calibrate data is batch_size*batch_nums. If batch_nums is None, use
                all data provided by sample_generator as calibrate data.
            scope(fluid.Scope, optional): The scope of the program, use it to load
                and save variables. If scope=None, get scope by global_scope().
            algo(str, optional): If algo=KL, use KL-divergenc method to
                get the more precise scale factor. If algo='direct', use
                abs_max methon to get the scale factor. Default is KL.
            quantizable_op_type(list[str], optional): List the type of ops
                that will be quantized. Default is ["conv2d", "depthwise_conv2d",
                "mul"].
            is_full_quantized(bool, optional): If set is_full_quantized as True,
                apply quantization to all supported quantizable op type. If set
                is_full_quantized as False, only apply quantization to the op type
                according to the input quantizable_op_type.
            is_use_cache_file(bool, optional): If set is_use_cache_file as False,
                all temp data will be saved in memory. If set is_use_cache_file as True,
                it will save temp data to disk. When the fp32 model is complex or
                the number of calibrate data is large, we should set is_use_cache_file
                as True. Defalut is False.
            cache_dir(str, optional): When is_use_cache_file is True, set cache_dir as
                the directory for saving temp data. Default is ./temp_post_training.
        Returns:
            None
        '''
        self._support_activation_quantize_type = [
            'range_abs_max', 'moving_average_abs_max', 'abs_max'
        ]
        self._support_weight_quantize_type = [
            'abs_max', 'channel_wise_abs_max'
        ]
        self._support_algo_type = ['KL', 'abs_max', 'min_max']
        self._support_quantize_op_type = \
            list(set(QuantizationTransformPass._supported_quantizable_op_type +
                AddQuantDequantPass._supported_quantizable_op_type))

        # Check inputs
        assert executor is not None, "The executor cannot be None."
        assert batch_size > 0, "The batch_size should be greater than 0."
        assert algo in self._support_algo_type, \
            "The algo should be KL, abs_max or min_max."

        self._executor = executor
        self._dataset = dataset
        self._batch_size = batch_size
        self._batch_nums = batch_nums
        self._scope = fluid.global_scope() if scope == None else scope
        self._algo = algo
        self._is_use_cache_file = is_use_cache_file
        self._cache_dir = cache_dir
        self._activation_bits = 8
        self._weight_bits = 8
        self._activation_quantize_type = 'range_abs_max'
        self._weight_quantize_type = 'channel_wise_abs_max'
        if self._is_use_cache_file and not os.path.exists(self._cache_dir):
            os.mkdir(self._cache_dir)

        if is_full_quantize:
            self._quantizable_op_type = self._support_quantize_op_type
        else:
            self._quantizable_op_type = quantizable_op_type
            for op_type in self._quantizable_op_type:
                assert op_type in self._support_quantize_op_type + \
                    AddQuantDequantPass._activation_type, \
                    op_type + " is not supported for quantization."

        self._place = self._executor.place
        self._program = program
        self._feed_list = list(inputs.values())
        self._fetch_list = list(outputs.values())
        self._data_loader = None

        self._out_scale_op_list = _out_scale_op_list
        self._bit_length = 8
        self._quantized_weight_var_name = set()
        self._quantized_act_var_name = set()
        self._sampling_data = {}
        self._quantized_var_kl_threshold = {}
        self._quantized_var_min = {}
        self._quantized_var_max = {}
        self._quantized_var_abs_max = {}

    def quantize(self):
        '''
        Quantize the fp32 model. Use calibrate data to calculate the scale factor of
        quantized variables, and inserts fake quant/dequant op to obtain the
        quantized model.
        Args:
            None
        Returns:
            the program of quantized model.
        '''
        self._load_model_data()
        self._collect_target_varnames()
        self._set_activation_persistable()
        batch_ct = 0
        for data in self._data_loader():
            batch_ct += 1
            if self._batch_nums and batch_ct >= self._batch_nums:
                break
        batch_id = 0
        logging.info("Start to run batch!")
        for data in self._data_loader():
            start = time.time()
            with fluid.scope_guard(self._scope):
                self._executor.run(program=self._program,
                                   feed=data,
                                   fetch_list=self._fetch_list,
                                   return_numpy=False)
            if self._algo == "KL":
                self._sample_data(batch_id)
            else:
                self._sample_threshold()
            end = time.time()
            logging.debug(
                '[Run batch data] Batch={}/{}, time_each_batch={} s.'.format(
                    str(batch_id + 1), str(batch_ct), str(end - start)))
            batch_id += 1
            if self._batch_nums and batch_id >= self._batch_nums:
                break
        logging.info("All run batch: ".format(batch_id))
        self._reset_activation_persistable()
        logging.info("Calculate scale factor ...")
        if self._algo == "KL":
            self._calculate_kl_threshold()
        logging.info("Update the program ...")
        if self._algo in ["KL", "abs_max"]:
            self._update_program()
        else:
            self._save_input_threhold()
        logging.info("Save ...")
        self._save_output_threshold()
        logging.info("Finish quant!")
        return self._program

    def save_quantized_model(self, save_model_path):
        '''
        Save the quantized model to the disk.
        Args:
            save_model_path(str): The path to save the quantized model
        Returns:
            None
        '''
        with fluid.scope_guard(self._scope):
            feed_vars_names = [var.name for var in self._feed_list]
            fluid.io.save_inference_model(
                dirname=save_model_path,
                feeded_var_names=feed_vars_names,
                target_vars=self._fetch_list,
                executor=self._executor,
                params_filename='__params__',
                main_program=self._program)

    def _load_model_data(self):
        '''
        Set data loader.
        '''
        feed_vars = [fluid.framework._get_var(var.name, self._program) \
            for var in self._feed_list]
        self._data_loader = fluid.io.DataLoader.from_generator(
            feed_list=feed_vars, capacity=3 * self._batch_size, iterable=True)
        self._data_loader.set_sample_list_generator(
            self._dataset.generator(
                self._batch_size, drop_last=True),
            places=self._place)

    def _calculate_kl_threshold(self):
        '''
        Calculate the KL threshold of quantized variables.
        '''
        assert self._algo == "KL", "The algo should be KL to calculate kl threshold."
        ct = 1
        # Abs_max threshold for weights
        for var_name in self._quantized_weight_var_name:
            start = time.time()
            weight_data = self._sampling_data[var_name]
            weight_threshold = None
            if self._weight_quantize_type == "abs_max":
                weight_threshold = np.max(np.abs(weight_data))
            elif self._weight_quantize_type == "channel_wise_abs_max":
                weight_threshold = []
                for i in range(weight_data.shape[0]):
                    abs_max_value = np.max(np.abs(weight_data[i]))
                    weight_threshold.append(abs_max_value)
            self._quantized_var_kl_threshold[var_name] = weight_threshold
            end = time.time()
            logging.debug(
                '[Calculate weight] Weight_id={}/{}, time_each_weight={} s.'.
                format(
                    str(ct),
                    str(len(self._quantized_weight_var_name)),
                    str(end - start)))
            ct += 1

        ct = 1
        # KL threshold for activations
        if self._is_use_cache_file:
            for var_name in self._quantized_act_var_name:
                start = time.time()
                sampling_data = []
                file_name = os.path.join(self._cache_dir, var_name)
                cache_dir, var_name_ = os.path.split(file_name)
                filenames = [f for f in os.listdir(cache_dir) \
                    if re.match(var_name_ + '_[0-9]+.npy', f)]
                for filename in filenames:
                    file_path = os.path.join(cache_dir, filename)
                    sampling_data.append(np.load(file_path))
                    os.remove(file_path)
                sampling_data = np.concatenate(sampling_data)
                self._quantized_var_kl_threshold[var_name] = \
                    self._get_kl_scaling_factor(np.abs(sampling_data))
                end = time.time()
                logging.debug(
                    '[Calculate activation] Activation_id={}/{}, time_each_activation={} s.'.
                    format(
                        str(ct),
                        str(len(self._quantized_act_var_name)),
                        str(end - start)))
                ct += 1
        else:
            for var_name in self._quantized_act_var_name:
                start = time.time()
                self._sampling_data[var_name] = np.concatenate(
                    self._sampling_data[var_name])
                self._quantized_var_kl_threshold[var_name] = \
                    self._get_kl_scaling_factor(np.abs(self._sampling_data[var_name]))
                end = time.time()
                logging.debug(
                    '[Calculate activation] Activation_id={}/{}, time_each_activation={} s.'.
                    format(
                        str(ct),
                        str(len(self._quantized_act_var_name)),
                        str(end - start)))
                ct += 1

    def _sample_data(self, iter):
        '''
        Sample the tensor data of quantized variables,
        applied in every iteration.
        '''
        assert self._algo == "KL", "The algo should be KL to sample data."
        for var_name in self._quantized_weight_var_name:
            if var_name not in self._sampling_data:
                var_tensor = _load_variable_data(self._scope, var_name)
                self._sampling_data[var_name] = var_tensor

        if self._is_use_cache_file:
            for var_name in self._quantized_act_var_name:
                var_tensor = _load_variable_data(self._scope, var_name)
                var_tensor = var_tensor.ravel()
                save_path = os.path.join(self._cache_dir,
                                         var_name + "_" + str(iter) + ".npy")
                save_dir, file_name = os.path.split(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_path, var_tensor)
        else:
            for var_name in self._quantized_act_var_name:
                if var_name not in self._sampling_data:
                    self._sampling_data[var_name] = []
                var_tensor = _load_variable_data(self._scope, var_name)
                var_tensor = var_tensor.ravel()
                self._sampling_data[var_name].append(var_tensor)


class PaddleXPostTrainingQuantizationV2(PostTrainingQuantization):
    def __init__(self,
                 executor,
                 dataset,
                 program,
                 inputs,
                 outputs,
                 batch_size=10,
                 batch_nums=None,
                 scope=None,
                 algo="KL",
                 quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
                 is_full_quantize=False,
                 activation_bits=8,
                 weight_bits=8,
                 activation_quantize_type='range_abs_max',
                 weight_quantize_type='channel_wise_abs_max',
                 optimize_model=False,
                 is_use_cache_file=False,
                 cache_dir="./temp_post_training"):
        '''
        Constructor.

        Args:
            executor(fluid.Executor): The executor to load, run and save the
                quantized model.
            dataset(Python Iterator): The data Reader.
            program(fluid.Program): The paddle program, save the parameters for model.
            inputs(dict): The input of prigram.
            outputs(dict): The output of program.

            scope(fluid.Scope, optional): The scope of the program, use it to load
                and save variables. If scope=None, get scope by global_scope().
            batch_size(int, optional): The batch size of DataLoader. Default is 10.
            batch_nums(int, optional): If batch_nums is not None, the number of
                calibrate data is batch_size*batch_nums. If batch_nums is None, use
                all data provided by sample_generator as calibrate data.
            algo(str, optional): If algo='KL', use KL-divergenc method to
                get the KL threshold for quantized activations and get the abs_max
                value for quantized weights. If algo='abs_max', get the abs max
                value for activations and weights. If algo= 'min_max', get the min
                and max value for quantized activations and weights. Default is KL.
            quantizable_op_type(list[str], optional): List the type of ops
                that will be quantized. Default is ["conv2d", "depthwise_conv2d",
                "mul"].
            is_full_quantized(bool, optional): If set is_full_quantized as True,
                apply quantization to all supported quantizable op type. If set
                is_full_quantized as False, only apply quantization to the op type
                according to the input quantizable_op_type.
            activation_bits(int): quantization bit number for activation.
            weight_bits(int, optional): quantization bit number for weights.
            activation_quantize_type(str): quantization type for activation,
                now support 'range_abs_max', 'moving_average_abs_max' and 'abs_max'.
                This param only specifies the fake ops in saving quantized model.
                If it is 'range_abs_max' or 'moving_average_abs_max', we save the scale
                obtained by post training quantization in fake ops. Note that, if it
                is 'abs_max', the scale will not be saved in fake ops.
            weight_quantize_type(str): quantization type for weights,
                support 'abs_max' and 'channel_wise_abs_max'. This param only specifies
                the fake ops in saving quantized model, and we save the scale obtained
                by post training quantization in fake ops. Compared to 'abs_max',
                the model accuracy is usually higher when it is 'channel_wise_abs_max'.
            optimize_model(bool, optional): If set optimize_model as True, it applies
                some passes to the model before quantization, and it supports
                `conv2d/depthwise_conv2d + bn` pass so far. Some targets require the
                weights are quantized by tensor-wise method, which means the weights
                scale for all channel are the same. However, if fuse
                `conv2d/depthwise_conv2d + bn`, the weights scale for all channel will
                be different. In address this problem, fuse the pattern before
                quantization. Default False.
            is_use_cache_file(bool, optional): This param is deprecated.
            cache_dir(str, optional): This param is deprecated.
        Returns:
            None

        '''

        self._support_activation_quantize_type = [
            'range_abs_max', 'moving_average_abs_max', 'abs_max'
        ]
        self._support_weight_quantize_type = [
            'abs_max', 'channel_wise_abs_max'
        ]
        self._support_algo_type = ['KL', 'abs_max', 'min_max']
        self._dynamic_quantize_op_type = ['lstm']
        self._support_quantize_op_type = \
            list(set(QuantizationTransformPass._supported_quantizable_op_type +
                AddQuantDequantPass._supported_quantizable_op_type +
                self._dynamic_quantize_op_type))

        # Check inputs
        assert executor is not None, "The executor cannot be None."

        assert batch_size > 0, "The batch_size should be greater than 0."
        assert algo in self._support_algo_type, \
            "The algo should be KL, abs_max or min_max."
        assert activation_quantize_type in self._support_activation_quantize_type, \
            "The activation_quantize_type ({}) should in ({}).".format(
            activation_quantize_type, self._support_activation_quantize_type)
        assert weight_quantize_type in self._support_weight_quantize_type, \
            "The weight_quantize_type ({}) shoud in ({}).".format(
            weight_quantize_type, self._support_weight_quantize_type)

        # Save input params
        self._executor = executor
        self._dataset = dataset
        self._scope = fluid.global_scope() if scope == None else scope
        self._batch_size = batch_size
        self._batch_nums = batch_nums
        self._algo = algo
        self._activation_bits = activation_bits
        self._weight_bits = weight_bits
        self._activation_quantize_type = activation_quantize_type
        self._weight_quantize_type = weight_quantize_type
        self._is_full_quantize = is_full_quantize
        if is_full_quantize:
            self._quantizable_op_type = self._support_quantize_op_type
        else:
            self._quantizable_op_type = quantizable_op_type
            for op_type in self._quantizable_op_type:
                assert op_type in self._support_quantize_op_type, \
                    op_type + " is not supported for quantization."
        self._optimize_model = optimize_model

        # Define variables
        self._place = self._executor.place
        self._program = program
        self._feed_list = [var.name for var in inputs.values()]
        self._fetch_list = list(outputs.values())
        self._data_loader = None

        self._out_scale_op_list = _out_scale_op_list
        self._quantized_weight_var_name = set()
        self._quantized_act_var_name = set()
        self._weight_op_pairs = {}
        # The vars for alog = KL
        self._sampling_act_abs_min_max = {}
        self._sampling_act_histogram = {}
        self._sampling_data = {}
        self._quantized_var_kl_threshold = {}
        self._histogram_bins = 2048
        # The vars for algo = min_max
        self._quantized_var_min = {}
        self._quantized_var_max = {}
        # The vars for algo = abs_max
        self._quantized_var_abs_max = {}

    def _load_model_data(self):
        '''
        Set data loader.
        '''
        logging.info("Set data loader ...")
        if self._program.num_blocks > 1:
            _logger.error("The post training quantization requires that the "
                          "program only has one block.")

        if self._optimize_model:
            self._optimize_fp32_model()

        feed_vars = [fluid.framework._get_var(var_name, self._program) \
            for var_name in self._feed_list]
        self._data_loader = fluid.io.DataLoader.from_generator(
            feed_list=feed_vars, capacity=3 * self._batch_size, iterable=True)
        self._data_loader.set_sample_list_generator(
            self._dataset.generator(
                self._batch_size, drop_last=True),
            places=self._place)

    def save_quantized_model(self, save_model_path):
        '''
        Save the quantized model to the disk.

        Args:
            save_model_path(str): The path to save the quantized model.
        Returns:
            None
        '''
        with fluid.scope_guard(self._scope):
            fluid.io.save_inference_model(
                dirname=save_model_path,
                model_filename='__model__',
                params_filename='__params__',
                feeded_var_names=self._feed_list,
                target_vars=self._fetch_list,
                executor=self._executor,
                main_program=self._program)
        logging.info("The quantized model is saved in " + save_model_path)
