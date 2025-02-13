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

from typing import Union, Tuple, List, Dict, Any, Iterator
import os
import shutil
from pathlib import Path
import lazy_paddle as paddle
import numpy as np

from ....utils.flags import DEBUG, FLAGS_json_format_model, USE_PIR_TRT
from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption


def collect_trt_shapes(
    model_file, model_params, gpu_id, shape_range_info_path, trt_dynamic_shapes
):
    config = paddle.inference.Config(model_file, model_params)
    config.enable_use_gpu(100, gpu_id)
    min_arrs, opt_arrs, max_arrs = {}, {}, {}
    for name, candidate_shapes in trt_dynamic_shapes.items():
        min_shape, opt_shape, max_shape = candidate_shapes
        min_arrs[name] = np.ones(min_shape, dtype=np.float32)
        opt_arrs[name] = np.ones(opt_shape, dtype=np.float32)
        max_arrs[name] = np.ones(max_shape, dtype=np.float32)

    config.collect_shape_range_info(shape_range_info_path)
    predictor = paddle.inference.create_predictor(config)
    # opt_arrs would be used twice to simulate the most common situations
    for arrs in [min_arrs, opt_arrs, opt_arrs, max_arrs]:
        for name, arr in arrs.items():
            input_handler = predictor.get_input_handle(name)
            input_handler.reshape(arr.shape)
            input_handler.copy_from_cpu(arr)
        predictor.run()


def convert_trt(mode, pp_model_path, trt_save_path, trt_dynamic_shapes):
    from lazy_paddle.tensorrt.export import (
        Input,
        TensorRTConfig,
        convert,
        PrecisionMode,
    )

    precision_map = {
        "trt_int8": PrecisionMode.INT8,
        "trt_fp32": PrecisionMode.FP32,
        "trt_fp16": PrecisionMode.FP16,
    }
    trt_inputs = []
    for name, candidate_shapes in trt_dynamic_shapes.items():
        min_shape, opt_shape, max_shape = candidate_shapes
        trt_input = Input(
            min_input_shape=min_shape,
            optim_input_shape=opt_shape,
            max_input_shape=max_shape,
        )
        trt_inputs.append(trt_input)

    # Create TensorRTConfig
    trt_config = TensorRTConfig(inputs=trt_inputs)
    trt_config.precision_mode = precision_map[mode]
    trt_config.save_model_dir = trt_save_path
    convert(pp_model_path, trt_config)


class Copy2GPU:

    def __init__(self, input_handlers):
        super().__init__()
        self.input_handlers = input_handlers

    def __call__(self, x):
        for idx in range(len(x)):
            self.input_handlers[idx].reshape(x[idx].shape)
            self.input_handlers[idx].copy_from_cpu(x[idx])


class Copy2CPU:

    def __init__(self, output_handlers):
        super().__init__()
        self.output_handlers = output_handlers

    def __call__(self):
        output = []
        for out_tensor in self.output_handlers:
            batch = out_tensor.copy_to_cpu()
            output.append(batch)
        return output


class Infer:

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def __call__(self):
        self.predictor.run()


class StaticInfer:
    """Predictor based on Paddle Inference"""

    def __init__(
        self, model_dir: str, model_prefix: str, option: PaddlePredictorOption
    ) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.model_prefix = model_prefix
        self._update_option(option)

    def _update_option(self, option: PaddlePredictorOption) -> None:
        if self.option and option == self.option:
            return
        self._option = option
        self._reset()

    @property
    def option(self) -> PaddlePredictorOption:
        return self._option if hasattr(self, "_option") else None

    @option.setter
    def option(self, option: Union[None, PaddlePredictorOption]) -> None:
        if option:
            self._update_option(option)

    def _reset(self) -> None:
        logging.debug(f"Env: {self.option}")
        (
            predictor,
            input_handlers,
            output_handlers,
        ) = self._create()
        self.copy2gpu = Copy2GPU(input_handlers)
        self.copy2cpu = Copy2CPU(output_handlers)
        self.infer = Infer(predictor)
        self.option.changed = False

    def _create(
        self,
    ) -> Tuple[
        "paddle.base.libpaddle.PaddleInferPredictor",
        "paddle.base.libpaddle.PaddleInferTensor",
        "paddle.base.libpaddle.PaddleInferTensor",
    ]:
        """_create"""
        from lazy_paddle.inference import Config, create_predictor

        if FLAGS_json_format_model:
            model_file = (self.model_dir / f"{self.model_prefix}.json").as_posix()
        # when FLAGS_json_format_model is not set, use inference.json if exist, otherwise inference.pdmodel
        else:
            model_file = self.model_dir / f"{self.model_prefix}.json"
            if model_file.exists():
                model_file = model_file.as_posix()
            # default by `pdmodel` suffix
            else:
                model_file = (
                    self.model_dir / f"{self.model_prefix}.pdmodel"
                ).as_posix()
        params_file = (self.model_dir / f"{self.model_prefix}.pdiparams").as_posix()

        # for TRT
        if self.option.run_mode.startswith("trt"):
            assert self.option.device == "gpu"
            if not USE_PIR_TRT:
                if self.option.shape_info_filename is None:
                    shape_range_info_path = (
                        self.model_dir / "shape_range_info.pbtxt"
                    ).as_posix()
                else:
                    shape_range_info_path = self.option.shape_info_filename
                if not os.path.exists(shape_range_info_path):
                    logging.info(
                        f"Dynamic shape info is collected into: {shape_range_info_path}"
                    )
                    collect_trt_shapes(
                        model_file,
                        params_file,
                        self.option.device_id,
                        shape_range_info_path,
                        self.option.trt_dynamic_shapes,
                    )
                else:
                    logging.info(
                        f"A dynamic shape info file ( {shape_range_info_path} ) already exists. No need to collect again."
                    )
                self.option.shape_info_filename = shape_range_info_path
            else:
                trt_save_path = (
                    Path(self.model_dir) / "trt" / self.model_prefix
                ).as_posix()
                pp_model_path = (Path(self.model_dir) / self.model_prefix).as_posix()
                convert_trt(
                    self.option.run_mode,
                    pp_model_path,
                    trt_save_path,
                    self.option.trt_dynamic_shapes,
                )
                model_file = trt_save_path + ".json"
                params_file = trt_save_path + ".pdiparams"

        config = Config(model_file, params_file)
        if self.option.device == "gpu":
            config.exp_disable_mixed_precision_ops({"feed", "fetch"})
            config.enable_use_gpu(100, self.option.device_id)
            if not self.option.run_mode.startswith("trt"):
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self.option.enable_new_ir)
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                config.set_optimization_level(3)
            # NOTE: The pptrt settings are not aligned with those of FD.
            else:
                if not USE_PIR_TRT:
                    precision_map = {
                        "trt_int8": Config.Precision.Int8,
                        "trt_fp32": Config.Precision.Float32,
                        "trt_fp16": Config.Precision.Half,
                    }
                    config.enable_tensorrt_engine(
                        workspace_size=(1 << 30) * self.option.batch_size,
                        max_batch_size=self.option.batch_size,
                        min_subgraph_size=self.option.min_subgraph_size,
                        precision_mode=precision_map[self.option.run_mode],
                        use_static=self.option.trt_use_static,
                        use_calib_mode=self.option.trt_calib_mode,
                    )
                    config.enable_tuned_tensorrt_dynamic_shape(
                        self.option.shape_info_filename, True
                    )
        elif self.option.device == "npu":
            config.enable_custom_device("npu")
        elif self.option.device == "xpu":
            pass
        elif self.option.device == "mlu":
            config.enable_custom_device("mlu")
        elif self.option.device == "dcu":
            config.enable_use_gpu(100, self.option.device_id)
            # XXX: is_compiled_with_rocm() must be True on dcu platform ?
            if paddle.is_compiled_with_rocm():
                # Delete unsupported passes in dcu
                config.delete_pass("conv2d_add_act_fuse_pass")
                config.delete_pass("conv2d_add_fuse_pass")
        else:
            assert self.option.device == "cpu"
            config.disable_gpu()
            if "mkldnn" in self.option.run_mode:
                try:
                    config.enable_mkldnn()
                    if "bf16" in self.option.run_mode:
                        config.enable_mkldnn_bfloat16()
                except Exception as e:
                    logging.warning(
                        "MKL-DNN is not available. We will disable MKL-DNN."
                    )
                config.set_mkldnn_cache_capacity(-1)
            else:
                if hasattr(config, "disable_mkldnn"):
                    config.disable_mkldnn()
            config.set_cpu_math_library_num_threads(self.option.cpu_threads)

            if hasattr(config, "enable_new_ir"):
                config.enable_new_ir(self.option.enable_new_ir)
            if hasattr(config, "enable_new_executor"):
                config.enable_new_executor()
            config.set_optimization_level(3)

        config.enable_memory_optim()
        for del_p in self.option.delete_pass:
            config.delete_pass(del_p)

        # Disable paddle inference logging
        if not DEBUG:
            config.disable_glog_info()

        predictor = create_predictor(config)

        # Get input and output handlers
        input_names = predictor.get_input_names()
        input_names.sort()
        input_handlers = []
        output_handlers = []
        for input_name in input_names:
            input_handler = predictor.get_input_handle(input_name)
            input_handlers.append(input_handler)
        output_names = predictor.get_output_names()
        for output_name in output_names:
            output_handler = predictor.get_output_handle(output_name)
            output_handlers.append(output_handler)
        return predictor, input_handlers, output_handlers

    def __call__(self, x) -> List[Any]:
        if self.option.changed:
            self._reset()
        self.copy2gpu(x)
        self.infer()
        pred = self.copy2cpu()
        return pred

    @property
    def benchmark(self):
        return {
            "Copy2GPU": self.copy2gpu,
            "Infer": self.infer,
            "Copy2CPU": self.copy2cpu,
        }
