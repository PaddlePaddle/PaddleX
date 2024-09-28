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

from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption
from ..utils.mixin import PPEngineMixin
from ..base import BaseComponent


class BasePaddlePredictor(BaseComponent, PPEngineMixin):
    """Predictor based on Paddle Inference"""

    OUTPUT_KEYS = "pred"
    DEAULT_OUTPUTS = {"pred": "pred"}
    ENABLE_BATCH = True

    def __init__(self, model_dir, model_prefix, option):
        super().__init__()
        PPEngineMixin.__init__(self, option)
        self.model_dir = model_dir
        self.model_prefix = model_prefix
        self._is_initialized = False

    def _reset(self):
        if not self.option:
            self.option = PaddlePredictorOption()
        (
            self.predictor,
            self.inference_config,
            self.input_names,
            self.input_handlers,
            self.output_handlers,
        ) = self._create()
        self._is_initialized = True
        logging.debug(f"Env: {self.option}")

    def _create(self):
        """_create"""
        from lazy_paddle.inference import Config, create_predictor

        use_pir = (
            hasattr(paddle.framework, "use_pir_api") and paddle.framework.use_pir_api()
        )
        model_postfix = ".json" if use_pir else ".pdmodel"
        model_file = (self.model_dir / f"{self.model_prefix}{model_postfix}").as_posix()
        params_file = (self.model_dir / f"{self.model_prefix}.pdiparams").as_posix()
        config = Config(model_file, params_file)

        if self.option.device == "gpu":
            config.enable_use_gpu(200, self.option.device_id)
            if paddle.is_compiled_with_rocm():
                os.environ["FLAGS_conv_workspace_size_limit"] = "2000"
            elif hasattr(config, "enable_new_ir"):
                config.enable_new_ir(self.option.enable_new_ir)
        elif self.option.device == "npu":
            config.enable_custom_device("npu")
            os.environ["FLAGS_npu_jit_compile"] = "0"
            os.environ["FLAGS_use_stride_kernel"] = "0"
            os.environ["FLAGS_allocator_strategy"] = "auto_growth"
            os.environ["CUSTOM_DEVICE_BLACK_LIST"] = (
                "pad3d,pad3d_grad,set_value,set_value_with_tensor"
            )
            os.environ["FLAGS_npu_scale_aclnn"] = "True"
            os.environ["FLAGS_npu_split_aclnn"] = "True"
        elif self.option.device == "xpu":
            os.environ["BKCL_FORCE_SYNC"] = "1"
            os.environ["BKCL_TIMEOUT"] = "1800"
            os.environ["FLAGS_use_stride_kernel"] = "0"
        elif self.option.device == "mlu":
            config.enable_custom_device("mlu")
            os.environ["FLAGS_use_stride_kernel"] = "0"
        else:
            assert self.option.device == "cpu"
            config.disable_gpu()
            if hasattr(config, "enable_new_ir"):
                config.enable_new_ir(self.option.enable_new_ir)
            if hasattr(config, "enable_new_executor"):
                config.enable_new_executor(True)
            if "mkldnn" in self.option.run_mode:
                try:
                    config.enable_mkldnn()
                    config.set_cpu_math_library_num_threads(self.option.cpu_threads)
                    if "bf16" in self.option.run_mode:
                        config.enable_mkldnn_bfloat16()
                except Exception as e:
                    logging.warning(
                        "MKL-DNN is not available. We will disable MKL-DNN."
                    )

        precision_map = {
            "trt_int8": Config.Precision.Int8,
            "trt_fp32": Config.Precision.Float32,
            "trt_fp16": Config.Precision.Half,
        }
        if self.option.run_mode in precision_map.keys():
            config.enable_tensorrt_engine(
                workspace_size=(1 << 25) * self.option.batch_size,
                max_batch_size=self.option.batch_size,
                min_subgraph_size=self.option.min_subgraph_size,
                precision_mode=precision_map[self.option.run_mode],
                trt_use_static=self.option.trt_use_static,
                use_calib_mode=self.option.trt_calib_mode,
            )

            if self.option.shape_info_filename is not None:
                if not os.path.exists(self.option.shape_info_filename):
                    config.collect_shape_range_info(self.option.shape_info_filename)
                    logging.info(
                        f"Dynamic shape info is collected into: {self.option.shape_info_filename}"
                    )
                else:
                    logging.info(
                        f"A dynamic shape info file ( {self.option.shape_info_filename} ) already exists. \
No need to generate again."
                    )
                config.enable_tuned_tensorrt_dynamic_shape(
                    self.option.shape_info_filename, True
                )

        # Disable paddle inference logging
        config.disable_glog_info()
        for del_p in self.option.delete_pass:
            config.delete_pass(del_p)
        # Enable shared memory
        config.enable_memory_optim()
        config.switch_ir_optim(True)
        # Disable feed, fetch OP, needed by zero_copy_run
        config.switch_use_feed_fetch_ops(False)
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
        return predictor, config, input_names, input_handlers, output_handlers

    def get_input_names(self):
        """get input names"""
        return self.input_names

    def apply(self, **kwargs):
        if not self._is_initialized:
            self._reset()

        x = self.to_batch(**kwargs)
        for idx in range(len(x)):
            self.input_handlers[idx].reshape(x[idx].shape)
            self.input_handlers[idx].copy_from_cpu(x[idx])

        self.predictor.run()
        output = []
        for out_tensor in self.output_handlers:
            batch = out_tensor.copy_to_cpu()
            output.append(batch)
        return self.format_output(output)

    def format_output(self, pred):
        return [{"pred": res} for res in zip(*pred)]

    @abstractmethod
    def to_batch(self):
        raise NotImplementedError


class ImagePredictor(BasePaddlePredictor):

    INPUT_KEYS = "img"
    DEAULT_INPUTS = {"img": "img"}

    def to_batch(self, img):
        return [np.stack(img, axis=0).astype(dtype=np.float32, copy=False)]


class ImageDetPredictor(BasePaddlePredictor):
    INPUT_KEYS = [["img", "scale_factors"], ["img", "scale_factors", "img_size"]]
    OUTPUT_KEYS = [["boxes"], ["boxes", "masks"]]
    DEAULT_INPUTS = {"img": "img", "scale_factors": "scale_factors"}
    DEAULT_OUTPUTS = None

    def to_batch(self, img, scale_factors, img_size=None):
        scale_factors = [scale_factor[::-1] for scale_factor in scale_factors]
        if img_size is None:
            return [
                np.stack(img, axis=0).astype(dtype=np.float32, copy=False),
                np.stack(scale_factors, axis=0).astype(dtype=np.float32, copy=False),
            ]
        else:
            # img_size = [img_size[::-1] for img_size in img_size]
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
    DEAULT_INPUTS = {"ts": "ts"}

    def to_batch(self, ts):
        n = len(ts[0])
        x = [np.stack([lst[i] for lst in ts], axis=0) for i in range(n)]
        return x
