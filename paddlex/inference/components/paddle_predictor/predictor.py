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

import os
from abc import abstractmethod
import lazy_paddle as paddle
import numpy as np

from ....utils.flags import FLAGS_json_format_model
from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption
from ..base import BaseComponent


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


class Copy2GPU(BaseComponent):

    def __init__(self, input_handlers):
        super().__init__()
        self.input_handlers = input_handlers

    def apply(self, x):
        for idx in range(len(x)):
            self.input_handlers[idx].reshape(x[idx].shape)
            self.input_handlers[idx].copy_from_cpu(x[idx])


class Copy2CPU(BaseComponent):

    def __init__(self, output_handlers):
        super().__init__()
        self.output_handlers = output_handlers

    def apply(self):
        output = []
        for out_tensor in self.output_handlers:
            batch = out_tensor.copy_to_cpu()
            output.append(batch)
        return output


class Infer(BaseComponent):

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def apply(self):
        self.predictor.run()


class BasePaddlePredictor(BaseComponent):
    """Predictor based on Paddle Inference"""

    OUTPUT_KEYS = "pred"
    DEAULT_OUTPUTS = {"pred": "pred"}
    ENABLE_BATCH = True

    def __init__(self, model_dir, model_prefix, option):
        super().__init__()
        self.model_dir = model_dir
        self.model_prefix = model_prefix
        self._update_option(option)

    def _update_option(self, option):
        if option:
            if self.option and option == self.option:
                return
            self._option = option
            self._reset()

    @property
    def option(self):
        return self._option if hasattr(self, "_option") else None

    @option.setter
    def option(self, option):
        self._update_option(option)

    def _reset(self):
        if not self.option:
            self.option = PaddlePredictorOption()
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

    def _create(self):
        """_create"""
        from lazy_paddle.inference import Config, create_predictor

        model_postfix = ".json" if FLAGS_json_format_model else ".pdmodel"
        model_file = (self.model_dir / f"{self.model_prefix}{model_postfix}").as_posix()
        params_file = (self.model_dir / f"{self.model_prefix}.pdiparams").as_posix()
        config = Config(model_file, params_file)

        config.enable_memory_optim()
        if self.option.device in ("gpu", "dcu"):
            if self.option.device == "gpu":
                config.exp_disable_mixed_precision_ops({"feed", "fetch"})
            config.enable_use_gpu(100, self.option.device_id)
            if self.option.device == "gpu":
                # NOTE: The pptrt settings are not aligned with those of FD.
                precision_map = {
                    "trt_int8": Config.Precision.Int8,
                    "trt_fp32": Config.Precision.Float32,
                    "trt_fp16": Config.Precision.Half,
                }
                if self.option.run_mode in precision_map.keys():
                    config.enable_tensorrt_engine(
                        workspace_size=(1 << 30) * self.option.batch_size,
                        max_batch_size=self.option.batch_size,
                        min_subgraph_size=self.option.min_subgraph_size,
                        precision_mode=precision_map[self.option.run_mode],
                        use_static=self.option.trt_use_static,
                        use_calib_mode=self.option.trt_calib_mode,
                    )

                    if not os.path.exists(self.option.shape_info_filename):
                        logging.info(
                            f"Dynamic shape info is collected into: {self.option.shape_info_filename}"
                        )
                        collect_trt_shapes(
                            model_file,
                            params_file,
                            self.option.device_id,
                            self.option.shape_info_filename,
                            self.option.trt_dynamic_shapes,
                        )
                    else:
                        logging.info(
                            f"A dynamic shape info file ( {self.option.shape_info_filename} ) already exists. No need to collect again."
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
        elif self.option.device == "gcu":
            assert paddle.device.is_compiled_with_custom_device("gcu"), (
                "Args device cannot be set as gcu while your paddle "
                "is not compiled with gcu!"
            )
            config.enable_custom_device("gcu")
            from paddle_custom_device.gcu import passes as gcu_passes

            gcu_passes.setUp()
            name = "PaddleX_" + self.option.model_name
            if hasattr(config, "enable_new_ir") and self.option.enable_new_ir:
                config.enable_new_ir(True)
                config.enable_new_executor(True)
                kPirGcuPasses = gcu_passes.inference_passes(use_pir=True, name=name)
                config.enable_custom_passes(kPirGcuPasses, True)
            else:
                config.enable_new_ir(False)
                config.enable_new_executor(False)
                pass_builder = config.pass_builder()
                gcu_passes.append_passes_for_legacy_ir(pass_builder, name)
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

        # Disable paddle inference logging
        config.disable_glog_info()

        config.set_cpu_math_library_num_threads(self.option.cpu_threads)

        if not (self.option.device == "gpu" and self.option.run_mode.startswith("trt")):
            if self.option.device in ("cpu", "gpu"):
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self.option.enable_new_ir)
                config.set_optimization_level(3)
            if hasattr(config, "enable_new_executor"):
                config.enable_new_executor()

        for del_p in self.option.delete_pass:
            config.delete_pass(del_p)

        if self.option.device in ("gpu", "dcu"):
            if paddle.is_compiled_with_rocm():
                # Delete unsupported passes in dcu
                config.delete_pass("conv2d_add_act_fuse_pass")
                config.delete_pass("conv2d_add_fuse_pass")

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

    def apply(self, **kwargs):
        if self.option.changed:
            self._reset()
        batches = self.to_batch(**kwargs)
        self.copy2gpu.apply(batches)
        self.infer.apply()
        pred = self.copy2cpu.apply()
        return self.format_output(pred)

    @property
    def sub_cmps(self):
        return {
            "Copy2GPU": self.copy2gpu,
            "Infer": self.infer,
            "Copy2CPU": self.copy2cpu,
        }

    @abstractmethod
    def to_batch(self):
        raise NotImplementedError

    @abstractmethod
    def format_output(self, pred):
        return [{"pred": res} for res in zip(*pred)]


class ImagePredictor(BasePaddlePredictor):
    INPUT_KEYS = "img"
    OUTPUT_KEYS = "pred"
    DEAULT_INPUTS = {"img": "img"}
    DEAULT_OUTPUTS = {"pred": "pred"}

    def to_batch(self, img):
        return [np.stack(img, axis=0).astype(dtype=np.float32, copy=False)]

    def format_output(self, pred):
        return [{"pred": res} for res in zip(*pred)]


class ImageDetPredictor(BasePaddlePredictor):

    INPUT_KEYS = [
        ["img", "scale_factors"],
        ["img", "scale_factors", "img_size"],
        ["img", "img_size"],
    ]
    OUTPUT_KEYS = [["boxes"], ["boxes", "masks"]]
    DEAULT_INPUTS = {"img": "img", "scale_factors": "scale_factors"}
    DEAULT_OUTPUTS = None

    def to_batch(self, img, scale_factors=[[1.0, 1.0]], img_size=None):
        scale_factors = [scale_factor[::-1] for scale_factor in scale_factors]
        if img_size is None:
            return [
                np.stack(img, axis=0).astype(dtype=np.float32, copy=False),
                np.stack(scale_factors, axis=0).astype(dtype=np.float32, copy=False),
            ]
        else:
            img_size = [img_size[::-1] for img_size in img_size]
            return [
                np.stack(img_size, axis=0).astype(dtype=np.float32, copy=False),
                np.stack(img, axis=0).astype(dtype=np.float32, copy=False),
                np.stack(scale_factors, axis=0).astype(dtype=np.float32, copy=False),
            ]

    def format_output(self, pred):
        box_idx_start = 0
        pred_box = []

        if len(pred) == 4:
            # Adapt to SOLOv2
            pred_class_id = []
            pred_mask = []
            pred_class_id.append([pred[1], pred[2]])
            pred_mask.append(pred[3])
            return [
                {
                    "class_id": np.array(pred_class_id[i]),
                    "masks": np.array(pred_mask[i]),
                }
                for i in range(len(pred_class_id))
            ]

        if len(pred) == 3:
            # Adapt to Instance Segmentation
            pred_mask = []
        for idx in range(len(pred[1])):
            np_boxes_num = pred[1][idx]
            box_idx_end = box_idx_start + np_boxes_num
            np_boxes = pred[0][box_idx_start:box_idx_end]
            pred_box.append(np_boxes)
            if len(pred) == 3:
                np_masks = pred[2][box_idx_start:box_idx_end]
                pred_mask.append(np_masks)
            box_idx_start = box_idx_end

        if len(pred) == 3:
            return [
                {"boxes": np.array(pred_box[i]), "masks": np.array(pred_mask[i])}
                for i in range(len(pred_box))
            ]
        else:
            return [{"boxes": np.array(res)} for res in pred_box]


class TSPPPredictor(BasePaddlePredictor):

    INPUT_KEYS = "ts"
    OUTPUT_KEYS = "pred"
    DEAULT_INPUTS = {"ts": "ts"}
    DEAULT_OUTPUTS = {"pred": "pred"}

    def to_batch(self, ts):
        n = len(ts[0])
        x = [np.stack([lst[i] for lst in ts], axis=0) for i in range(n)]
        return x

    def format_output(self, pred):
        return [{"pred": res} for res in zip(*pred)]
