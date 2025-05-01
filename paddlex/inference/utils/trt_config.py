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

from collections import defaultdict

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
        from paddle.inference import PrecisionType

        return {
            "trt_int8": PrecisionType.Int8,
            "trt_fp32": PrecisionType.Float32,
            "trt_fp16": PrecisionType.Half,
        }


class PIR_TRT_PRECISION_MAP_CLASS(LazyLoadDict):
    def _load(self):
        from paddle.tensorrt.export import PrecisionMode

        return {
            "trt_int8": PrecisionMode.INT8,
            "trt_fp32": PrecisionMode.FP32,
            "trt_fp16": PrecisionMode.FP16,
        }


############ old ir trt ############
OLD_IR_TRT_PRECISION_MAP = OLD_IR_TRT_PRECISION_MAP_CLASS()

OLD_IR_TRT_CFG_DEFAULT_SETTING = {
    "workspace_size": 1 << 30,
    "max_batch_size": 32,
    "min_subgraph_size": 3,
    "use_static": True,
    "use_calib_mode": False,
}

OLD_IR_TRT_CFG_SETTING = {
    "SegFormer-B0": {
        "enable_tensorrt_engine": {
            **OLD_IR_TRT_CFG_DEFAULT_SETTING,
            "workspace_size": 1 << 32,
        }
    },
    "SegFormer-B1": {
        "enable_tensorrt_engine": {
            **OLD_IR_TRT_CFG_DEFAULT_SETTING,
            "workspace_size": 1 << 32,
        }
    },
    "SegFormer-B2": {
        "enable_tensorrt_engine": {
            **OLD_IR_TRT_CFG_DEFAULT_SETTING,
            "workspace_size": 1 << 32,
        }
    },
    "SegFormer-B3": {
        "enable_tensorrt_engine": {
            **OLD_IR_TRT_CFG_DEFAULT_SETTING,
            "workspace_size": 1 << 32,
        }
    },
    "SegFormer-B4": {
        "enable_tensorrt_engine": {
            **OLD_IR_TRT_CFG_DEFAULT_SETTING,
            "workspace_size": 1 << 32,
        }
    },
    "SegFormer-B5": {
        "enable_tensorrt_engine": {
            **OLD_IR_TRT_CFG_DEFAULT_SETTING,
            "workspace_size": 1 << 32,
        }
    },
    "SLANeXt_wired": {
        "enable_tensorrt_engine": OLD_IR_TRT_CFG_DEFAULT_SETTING,
        # the exp_disable_tensorrt_ops() func don't support to be pass argument by keyword
        # therefore, using list instead of dict
        "exp_disable_tensorrt_ops": [
            [
                "linear_0.tmp_0",
                "linear_4.tmp_0",
                "linear_12.tmp_0",
                "linear_16.tmp_0",
                "linear_24.tmp_0",
                "linear_28.tmp_0",
                "linear_36.tmp_0",
                "linear_40.tmp_0",
            ]
        ],
    },
    "SLANeXt_wireless": {
        "enable_tensorrt_engine": OLD_IR_TRT_CFG_DEFAULT_SETTING,
        "exp_disable_tensorrt_ops": [
            [
                "linear_0.tmp_0",
                "linear_4.tmp_0",
                "linear_12.tmp_0",
                "linear_16.tmp_0",
                "linear_24.tmp_0",
                "linear_28.tmp_0",
                "linear_36.tmp_0",
                "linear_40.tmp_0",
            ]
        ],
    },
    "PP-YOLOE_seg-S": {
        "enable_tensorrt_engine": OLD_IR_TRT_CFG_DEFAULT_SETTING,
        "exp_disable_tensorrt_ops": [
            ["bilinear_interp_v2_1.tmp_0", "bilinear_interp_v2_1.tmp_0_slice_0"]
        ],
    },
    "TiDE": {
        "enable_tensorrt_engine": OLD_IR_TRT_CFG_DEFAULT_SETTING,
        "exp_disable_tensorrt_ops": [
            [
                "reshape2_3.tmp_0",
                "reshape2_2.tmp_0",
                "reshape2_1.tmp_0",
                "reshape2_0.tmp_0",
            ]
        ],
    },
    "Nonstationary": {
        "enable_tensorrt_engine": OLD_IR_TRT_CFG_DEFAULT_SETTING,
        "exp_disable_tensorrt_ops": [
            [
                "reshape2_13.tmp_0",
            ]
        ],
    },
    "ch_SVTRv2_rec": {
        "enable_tensorrt_engine": OLD_IR_TRT_CFG_DEFAULT_SETTING,
        "exp_disable_tensorrt_ops": [
            [
                "reshape2_3.tmp_0",
                "reshape2_5.tmp_0",
                "reshape2_7.tmp_0",
                "reshape2_9.tmp_0",
                "reshape2_11.tmp_0",
                "reshape2_13.tmp_0",
                "reshape2_15.tmp_0",
                "reshape2_17.tmp_0",
                "reshape2_19.tmp_0",
                "reshape2_28.tmp_0",
                "reshape2_42.tmp_0",
                "reshape2_47.tmp_0",
                "layer_norm_15.tmp_2",
                "layer_norm_13.tmp_2",
            ]
        ],
    },
    "PP-YOLOE_plus_SOD-largesize-L": {
        "enable_tensorrt_engine": OLD_IR_TRT_CFG_DEFAULT_SETTING,
        "exp_disable_tensorrt_ops": [
            [
                "conv2d",
                "fused_conv2d_add_act",
                "swish",
                "reduce_mean",
                "softmax",
                "layer_norm",
                "gelu",
            ]
        ],
    },
}

DISABLE_TRT_HALF_OPS_CONFIG = {
    "ConvNeXt_tiny": {"layer_norm"},
    "ConvNeXt_small": {"layer_norm"},
    "ConvNeXt_base_224": {"layer_norm"},
    "ConvNeXt_large_224": {"layer_norm"},
    "ConvNeXt_base_384": {"layer_norm"},
    "ConvNeXt_large_384": {"layer_norm"},
    "PP-HGNetV2-B3": {"softmax"},
    "MobileNetV1_x0_5": {"fused_conv2d_add_act"},
    "SeaFormer_small": {"fused_conv2d_add_act"},
    "SeaFormer_tiny": {"fused_conv2d_add_act"},
    "PP-OCRv4_mobile_seal_det": {
        "fused_conv2d_add_act",
        "softmax",
        "conv2d",
        "multiply",
    },
    "PicoDet_LCNet_x2_5_face": {
        "fused_conv2d_add_act",
        "softmax",
        "elementwise_mul",
        "matrix_multiply",
    },
    "PP-YOLOE_plus_SOD-S": {
        "fused_conv2d_add_act",
        "softmax",
        "conv2d",
        "elementwise_mul",
        "matrix_multiply",
    },
    "BlazeFace-FPN-SSH": {"fused_conv2d_add_act"},
    "PP-YOLOE_plus-S_face": {"fused_conv2d_add_act", "conv2d", "multiply"},
    "PP-ShiTuV2_det": {
        "conv2d",
        "depthwise_conv2d",
        "fused_conv2d_add_act",
        "matrix_multiply",
    },
    "RT-DETR-H_layout_3cls": {
        "fused_conv2d_add_act",
        "elementwise_mul",
        "elementwise_add",
        "elementwise_div",
        "matrix_multiply",
        "layer_norm",
    },
    "DETR-R50": {
        "fused_conv2d_add_act",
        "elementwise_mul",
        "elementwise_add",
        "elementwise_div",
        "matrix_multiply",
        "layer_norm",
    },
    "RT-DETR-R50": {
        "fused_conv2d_add_act",
        "elementwise_mul",
        "elementwise_add",
        "elementwise_div",
        "matrix_multiply",
        "layer_norm",
    },
    "YOLOX-M": {"fused_conv2d_add_act", "elementwise_mul", "elementwise_add", "scale"},
    "YOLOv3-MobileNetV3": {
        "fused_conv2d_add_act",
        "elementwise_mul",
        "elementwise_add",
        "depthwise_conv2d",
        "elementwise_div",
    },
    "PP-OCRv4_server_det": {"fused_conv2d_add_act", "conv2d"},
}

############ pir trt ############
PIR_TRT_PRECISION_MAP = PIR_TRT_PRECISION_MAP_CLASS()

PIR_TRT_CFG_SETTING = {
    "PP-YOLOE_plus_SOD-largesize-L": {
        "workspace_size": 1 << 32,
        "disable_ops": [
            "pd_op.conv2d",
            "pd_op.fused_conv2d_add_act",
            "pd_op.swish",
            "pd_op.mean",
            "pd_op.softmax",
            "pd_op.layer_norm",
            "pd_op.gelu",
        ],
    },
    "SLANeXt_wired": {"disable_ops": ["pd_op.slice"]},
    "SLANeXt_wireless": {"disable_ops": ["pd_op.slice"]},
    "DETR-R50": {
        "optimization_level": 4,
        "workspace_size": 1 << 32,
        "ops_run_float": {"pd_op.matmul", "pd_op.conv2d", "pd_op.fused_conv2d_add_act"},
    },
    "SegFormer-B0": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B1": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B2": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B3": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B4": {"optimization_level": 4, "workspace_size": 1 << 32},
    "SegFormer-B5": {"optimization_level": 4, "workspace_size": 1 << 32},
    "LaTeX_OCR_rec": {"disable_ops": ["pd_op.slice", "pd_op.reshape"]},
    "PP-YOLOE_seg-S": {
        "disable_ops": ["pd_op.slice", "pd_op.bilinear_interp"],
        "ops_run_float": {
            "pd_op.conv2d",
            "pd_op.fused_conv2d_add_act",
            "pd_op.conv2d_transpose",
            "pd_op.matmul",
        },
    },
    "PP-FormulaNet-L": {
        "disable_ops": ["pd_op.full_with_tensor"],
        "workspace_size": 2 << 32,
    },
    "PP-FormulaNet-S": {
        "disable_ops": ["pd_op.full_with_tensor"],
        "workspace_size": 1 << 32,
    },
    "ConvNeXt_tiny": {"ops_run_float": {"pd_op.layer_norm"}},
    "ConvNeXt_small": {"ops_run_float": {"pd_op.layer_norm"}},
    "ConvNeXt_base_224": {"ops_run_float": {"pd_op.layer_norm"}},
    "ConvNeXt_base_384": {"ops_run_float": {"pd_op.layer_norm"}},
    "ConvNeXt_large_224": {"ops_run_float": {"pd_op.layer_norm"}},
    "ConvNeXt_large_384": {"ops_run_float": {"pd_op.layer_norm"}},
    "PP-HGNetV2-B3": {"ops_run_float": {"pd_op.softmax"}},
    "BlazeFace-FPN-SSH": {"ops_run_float": {"pd_op.fused_conv2d_add_act"}},
    "PP-OCRv4_mobile_seal_det": {
        "ops_run_float": {
            "pd_op.fused_conv2d_add_act",
            "pd_op.softmax",
            "pd_op.multiply",
            "pd_op.conv2d",
        }
    },
    "PP-YOLOE_plus_SOD-S": {
        "ops_run_float": {
            "pd_op.fused_conv2d_add_act",
            "pd_op.softmax",
            "pd_op.conv2d",
            "pd_op.multiply",
            "pd_op.matmul",
        }
    },
    "PicoDet_LCNet_x2_5_face": {
        "ops_run_float": {
            "pd_op.fused_conv2d_add_act",
            "pd_op.softmax",
            "pd_op.conv2d",
            "pd_op.multiply",
            "pd_op.matmul",
        }
    },
    "PP-YOLOE_plus-S_face": {
        "ops_run_float": {
            "pd_op.fused_conv2d_add_act",
            "pd_op.multiply",
            "pd_op.conv2d",
        }
    },
    "PP-ShiTuV2_det": {
        "ops_run_float": {
            "pd_op.fused_conv2d_add_act",
            "pd_op.depthwise_conv2d",
            "pd_op.conv2d",
        }
    },
    "RT-DETR-H_layout_3cls": {
        "ops_run_float": {
            "pd_op.matmul",
            "pd_op.conv2d",
            "pd_op.depthwise_conv2d",
            "pd_op.fused_conv2d_add_act",
            "pd_op.batch_norm_",
        }
    },
    "RT-DETR-R50": {
        "ops_run_float": {"pd_op.matmul", "pd_op.conv2d", "pd_op.fused_conv2d_add_act"}
    },
    "YOLOX-M": {
        "ops_run_float": {
            "pd_op.multiply",
            "pd_op.conv2d",
            "pd_op.fused_conv2d_add_act",
        }
    },
    "YOLOv3-MobileNetV3": {
        "ops_run_float": {
            "pd_op.depthwise_conv2d",
            "pd_op.conv2d",
            "pd_op.fused_conv2d_add_act",
        }
    },
    "PP-OCRv4_server_det": {
        "ops_run_float": {"pd_op.conv2d", "pd_op.fused_conv2d_add_act"}
    },
    "PP-OCRv4_server_seal_det": {
        "ops_run_float": {"pd_op.conv2d", "pd_op.fused_conv2d_add_act"}
    },
    "PP-YOLOE_plus-M": {
        "ops_run_float": {"pd_op.conv2d", "pd_op.fused_conv2d_add_act"}
    },
}


if USE_PIR_TRT:
    TRT_PRECISION_MAP = PIR_TRT_PRECISION_MAP
    TRT_CFG_SETTING = defaultdict(dict, PIR_TRT_CFG_SETTING)
else:
    TRT_PRECISION_MAP = OLD_IR_TRT_PRECISION_MAP
    TRT_CFG_SETTING = defaultdict(
        lambda: {"enable_tensorrt_engine": OLD_IR_TRT_CFG_DEFAULT_SETTING},
        OLD_IR_TRT_CFG_SETTING,
    )
