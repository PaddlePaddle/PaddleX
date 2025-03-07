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

from collections import defaultdict
import lazy_paddle
from ...utils.flags import USE_PIR_TRT


class LazyLoadDict(dict):
    def __init__(self, *args, **kwargs):
        self._initialized = False
        super().__init__(*args, **kwargs)

    def _initialize(self):
        if not self._initialized:
            self.update(self._load())
            self._initialized = True

    def __getitem__(self, key):
        self._initialize()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._initialize()
        return super().__contains__(key)

    def _load(self):
        raise NotImplementedError


class OLD_IR_TRT_PRECISION_MAP_CLASS(LazyLoadDict):
    def _load(self):
        from lazy_paddle.inference import PrecisionType

        return {
            "trt_int8": PrecisionType.Int8,
            "trt_fp32": PrecisionType.Float32,
            "trt_fp16": PrecisionType.Half,
        }


class PIR_TRT_PRECISION_MAP_CLASS(LazyLoadDict):
    def _load(self):
        from lazy_paddle.tensorrt.export import PrecisionMode

        return {
            "trt_int8": PrecisionMode.INT8,
            "trt_fp32": PrecisionMode.FP32,
            "trt_fp16": PrecisionMode.FP16,
        }


############ old ir trt ############
OLD_IR_TRT_PRECISION_MAP = OLD_IR_TRT_PRECISION_MAP_CLASS()

OLD_IR_TRT_DEFAULT_CFG = {
    "workspace_size": 1 << 30,
    "max_batch_size": 32,
    "min_subgraph_size": 3,
    "use_static": True,
    "use_calib_mode": False,
}

OLD_IR_TRT_CFG = {
    "SegFormer-B3": {
        "enable_tensorrt_engine": {**OLD_IR_TRT_DEFAULT_CFG, "workspace_size": 1 << 31}
    },
    "SegFormer-B4": {
        "enable_tensorrt_engine": {**OLD_IR_TRT_DEFAULT_CFG, "workspace_size": 1 << 31}
    },
    "SegFormer-B5": {
        "enable_tensorrt_engine": {**OLD_IR_TRT_DEFAULT_CFG, "workspace_size": 1 << 31}
    },
    "SLANeXt_wired": {
        "enable_tensorrt_engine": OLD_IR_TRT_DEFAULT_CFG,
        "exp_disable_tensorrt_ops": {
            "ops": [
                "linear_0.tmp_0",
                "linear_4.tmp_0",
                "linear_12.tmp_0",
                "linear_16.tmp_0",
                "linear_24.tmp_0",
                "linear_28.tmp_0",
                "linear_36.tmp_0",
                "linear_40.tmp_0",
            ]
        },
    },
    "SLANeXt_wireless": {
        "enable_tensorrt_engine": OLD_IR_TRT_DEFAULT_CFG,
        "exp_disable_tensorrt_ops": {
            "ops": [
                "linear_0.tmp_0",
                "linear_4.tmp_0",
                "linear_12.tmp_0",
                "linear_16.tmp_0",
                "linear_24.tmp_0",
                "linear_28.tmp_0",
                "linear_36.tmp_0",
                "linear_40.tmp_0",
            ]
        },
    },
    "PP-YOLOE_seg-S": {
        "enable_tensorrt_engine": OLD_IR_TRT_DEFAULT_CFG,
        "exp_disable_tensorrt_ops": {
            "ops": ["bilinear_interp_v2_1.tmp_0", "bilinear_interp_v2_1.tmp_0_slice_0"]
        },
    },
}

############ pir trt ############
PIR_TRT_PRECISION_MAP = PIR_TRT_PRECISION_MAP_CLASS()

PIR_TRT_CFG = {
    "DETR-R50": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B0": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B1": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B2": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B3": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B4": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B5": {"optimization_level": 4, "workspace_size": 1 << 32},
    "LaTeX_OCR_rec": {"disable_ops": ["pd_op.slice"]},
    "PP-YOLOE_seg-S": {"disable_ops": ["pd_op.slice", "pd_op.bilinear_interp"]},
    "PP-FormulaNet-L": {
        "disable_ops": ["pd_op.full_with_tensor"],
        "workspace_size": 1 << 32,
    },
    "PP-FormulaNet-S": {
        "disable_ops": ["pd_op.full_with_tensor"],
        "workspace_size": 1 << 32,
    },
}


if USE_PIR_TRT:
    TRT_PRECISION_MAP = PIR_TRT_PRECISION_MAP
    TRT_CFG = defaultdict(dict, PIR_TRT_CFG)
else:
    TRT_PRECISION_MAP = OLD_IR_TRT_PRECISION_MAP
    TRT_CFG = defaultdict(
        lambda: {"enable_tensorrt_engine": OLD_IR_TRT_DEFAULT_CFG}, OLD_IR_TRT_CFG
    )
