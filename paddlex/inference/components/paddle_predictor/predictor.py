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

from ..base import BaseComponent
from ....utils import logging


class BasePaddlePredictor(BaseComponent):
    """Predictor based on Paddle Inference"""

    INPUT_KEYS = "batch_data"
    OUTPUT_KEYS = "pred"
    DEAULT_INPUTS = {"batch_data": "batch_data"}
    DEAULT_OUTPUTS = {"pred": "pred"}
    ENABLE_BATCH = True

    def __init__(self, model_dir, model_prefix, option):
        super().__init__()
        (
            self.predictor,
            self.inference_config,
            self.input_names,
            self.input_handlers,
            self.output_handlers,
        ) = self._create(model_dir, model_prefix, option)

    def _create(self, model_dir, model_prefix, option):
        """_create"""
        from lazy_paddle.inference import Config, create_predictor

        use_pir = (
            hasattr(paddle.framework, "use_pir_api") and paddle.framework.use_pir_api()
        )
        model_postfix = ".json" if use_pir else ".pdmodel"
        model_file = (model_dir / f"{model_prefix}{model_postfix}").as_posix()
        params_file = (model_dir / f"{model_prefix}.pdiparams").as_posix()
        config = Config(model_file, params_file)

        if option.device == "gpu":
            config.enable_use_gpu(200, option.device_id)
            if paddle.is_compiled_with_rocm():
                os.environ["FLAGS_conv_workspace_size_limit"] = "2000"
            elif hasattr(config, "enable_new_ir"):
                config.enable_new_ir(option.enable_new_ir)
        elif option.device == "npu":
            config.enable_custom_device("npu")
            os.environ["FLAGS_npu_jit_compile"] = "0"
            os.environ["FLAGS_use_stride_kernel"] = "0"
            os.environ["FLAGS_allocator_strategy"] = "auto_growth"
            os.environ["CUSTOM_DEVICE_BLACK_LIST"] = (
                "pad3d,pad3d_grad,set_value,set_value_with_tensor"
            )
            os.environ["FLAGS_npu_scale_aclnn"] = "True"
            os.environ["FLAGS_npu_split_aclnn"] = "True"
        elif option.device == "xpu":
            os.environ["BKCL_FORCE_SYNC"] = "1"
            os.environ["BKCL_TIMEOUT"] = "1800"
            os.environ["FLAGS_use_stride_kernel"] = "0"
        elif option.device == "mlu":
            config.enable_custom_device("mlu")
            os.environ["FLAGS_use_stride_kernel"] = "0"
        else:
            assert option.device == "cpu"
            config.disable_gpu()
            config.enable_new_ir(option.enable_new_ir)
            config.enable_new_executor(True)
            if "mkldnn" in option.run_mode:
                try:
                    config.enable_mkldnn()
                    config.set_cpu_math_library_num_threads(option.cpu_threads)
                    if "bf16" in option.run_mode:
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
        if option.run_mode in precision_map.keys():
            config.enable_tensorrt_engine(
                workspace_size=(1 << 25) * option.batch_size,
                max_batch_size=option.batch_size,
                min_subgraph_size=option.min_subgraph_size,
                precision_mode=precision_map[option.run_mode],
                trt_use_static=option.trt_use_static,
                use_calib_mode=option.trt_calib_mode,
            )

            if option.shape_info_filename is not None:
                if not os.path.exists(option.shape_info_filename):
                    config.collect_shape_range_info(option.shape_info_filename)
                    logging.info(
                        f"Dynamic shape info is collected into: {option.shape_info_filename}"
                    )
                else:
                    logging.info(
                        f"A dynamic shape info file ( {option.shape_info_filename} ) already exists. \
No need to generate again."
                    )
                config.enable_tuned_tensorrt_dynamic_shape(
                    option.shape_info_filename, True
                )

        # Disable paddle inference logging
        config.disable_glog_info()
        for del_p in option.delete_pass:
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
    DEAULT_INPUTS = {"img": "img"}

    def to_batch(self, img):
        return [np.stack(img, axis=0).astype(dtype=np.float32, copy=False)]


class ImageDetPredictor(BasePaddlePredictor):
    INPUT_KEYS = [["img", "scale_factors"], ["img", "scale_factors", "img_size"]]
    OUTPUT_KEYS = [["boxes"], ["boxes", "masks"]]
    DEAULT_INPUTS = {"img": "img", "scale_factors": "scale_factors"}
    DEAULT_OUTPUTS = {"boxes": "boxes"}

    def to_batch(self, img, scale_factors, img_size=None):
        scale_factors = [scale_factor[::-1] for scale_factor in scale_factors]
        if img_size is None:
            return [
                np.stack(img, axis=0).astype(dtype=np.float32, copy=False),
                np.stack(scale_factors, axis=0).astype(dtype=np.float32, copy=False),
            ]
        else:
            return [
                np.stack(img_size, axis=0).astype(dtype=np.float32, copy=False),
                np.stack(img, axis=0).astype(dtype=np.float32, copy=False),
                np.stack(scale_factors, axis=0).astype(dtype=np.float32, copy=False),
            ]

    def format_output(self, pred):
        box_idx_start = 0
        pred_box = []
        if len(pred) == 3:
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

        boxes = [{"boxes": np.array(res)} for res in pred_box]
        if len(pred) == 3:
            masks = [{"masks": np.array(res)} for res in pred_mask]
            return [{"boxes": boxes[0]["boxes"], "masks": masks[0]["masks"]}]
        else:
            return [{"boxes": np.array(res)} for res in pred_box]


class ImageInstanceSegPredictor(ImageDetPredictor):
    DEAULT_INPUTS = {
        "img": "img",
        "scale_factors": "scale_factors",
        "img_size": "img_size",
    }
    DEAULT_OUTPUTS = {"boxes": "boxes", "masks": "masks"}
