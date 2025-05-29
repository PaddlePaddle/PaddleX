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

import os
import os.path as osp

from ...base.register import register_model_info, register_suite_info
from .config import ClsConfig
from .model import ClsModel
from .runner import ClsRunner

REPO_ROOT_PATH = os.environ.get("PADDLE_PDX_PADDLECLAS_PATH")
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "configs"))

register_suite_info(
    {
        "suite_name": "Cls",
        "model": ClsModel,
        "runner": ClsRunner,
        "config": ClsConfig,
        "runner_root_path": REPO_ROOT_PATH,
    }
)

################ Models Using Universal Config ################
register_model_info(
    {
        "model_name": "SwinTransformer_tiny_patch4_window7_224",
        "suite": "Cls",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "SwinTransformer_tiny_patch4_window7_224.yaml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "SwinTransformer_small_patch4_window7_224",
        "suite": "Cls",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "SwinTransformer_small_patch4_window7_224.yaml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "SwinTransformer_base_patch4_window7_224",
        "suite": "Cls",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "SwinTransformer_base_patch4_window7_224.yaml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "SwinTransformer_base_patch4_window12_384",
        "suite": "Cls",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "SwinTransformer_base_patch4_window12_384.yaml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "SwinTransformer_large_patch4_window7_224",
        "suite": "Cls",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "SwinTransformer_large_patch4_window7_224.yaml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "SwinTransformer_large_patch4_window12_384",
        "suite": "Cls",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "SwinTransformer_large_patch4_window12_384.yaml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x0_25",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x0_25.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x0_35",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x0_35.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x0_5",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x0_5.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x0_75",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x0_75.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x1_0",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x1_0.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x1_0_doc_ori",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x1_0_doc_ori.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x1_5",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x1_5.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x2_0",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x2_0.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x2_5",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x2_5.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-LCNetV2_small",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNetV2_small.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-LCNetV2_base",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNetV2_base.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-LCNetV2_large",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNetV2_large.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "CLIP_vit_base_patch16_224",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "CLIP_vit_base_patch16_224.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "CLIP_vit_large_patch14_224",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "CLIP_vit_large_patch14_224.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-HGNet_tiny",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNet_tiny.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-HGNet_small",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNet_small.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-HGNet_base",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNet_base.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-HGNetV2-B0",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNetV2-B0.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-HGNetV2-B1",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNetV2-B1.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-HGNetV2-B2",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNetV2-B2.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-HGNetV2-B3",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNetV2-B3.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-HGNetV2-B4",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNetV2-B4.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-HGNetV2-B5",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNetV2-B5.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-HGNetV2-B6",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNetV2-B6.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ResNet18",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet18.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ResNet18_vd",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet18_vd.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ResNet34",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet34.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ResNet34_vd",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet34_vd.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ResNet50",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet50.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ResNet50_vd",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet50_vd.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ResNet101",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet101.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ResNet101_vd",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet101_vd.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ResNet152",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet152.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ResNet152_vd",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet152_vd.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ResNet200_vd",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet200_vd.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV1_x0_25",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV1_x0_25.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "MobileNetV1_x0_5",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV1_x0_5.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "MobileNetV1_x0_75",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV1_x0_75.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "MobileNetV1_x1_0",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV1_x1_0.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "MobileNetV2_x0_25",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV2_x0_25.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV2_x0_5",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV2_x0_5.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV2_x1_0",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV2_x1_0.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV2_x1_5",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV2_x1_5.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV2_x2_0",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV2_x2_0.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV3_large_x0_35",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV3_large_x0_35.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV3_large_x0_5",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV3_large_x0_5.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV3_large_x0_75",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV3_large_x0_75.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV3_large_x1_0",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV3_large_x1_0.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV3_large_x1_25",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV3_large_x1_25.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV3_small_x0_35",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV3_small_x0_35.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV3_small_x0_5",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV3_small_x0_5.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV3_small_x0_75",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV3_small_x0_75.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV3_small_x1_0",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV3_small_x1_0.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV3_small_x1_25",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV3_small_x1_25.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV4_conv_small",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV4_conv_small.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV4_conv_medium",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV4_conv_medium.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV4_conv_large",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV4_conv_large.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV4_hybrid_medium",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV4_hybrid_medium.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "MobileNetV4_hybrid_large",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileNetV4_hybrid_large.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ConvNeXt_tiny",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ConvNeXt_tiny.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ConvNeXt_small",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ConvNeXt_small.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ConvNeXt_base_224",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ConvNeXt_base_224.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ConvNeXt_base_384",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ConvNeXt_base_384.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ConvNeXt_large_224",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ConvNeXt_large_224.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ConvNeXt_large_384",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ConvNeXt_large_384.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "StarNet-S1",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "StarNet-S1.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "StarNet-S2",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "StarNet-S2.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "StarNet-S3",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "StarNet-S3.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "StarNet-S4",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "StarNet-S4.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "FasterNet-L",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterNet-L.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "FasterNet-M",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterNet-M.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "FasterNet-S",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterNet-S.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "FasterNet-T0",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterNet-T0.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "FasterNet-T1",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterNet-T1.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "FasterNet-T2",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterNet-T2.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x1_0_ML",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x1_0_ML.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "ResNet50_ML",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet50_ML.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "PP-HGNetV2-B0_ML",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNetV2-B0_ML.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "PP-HGNetV2-B4_ML",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNetV2-B4_ML.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "PP-HGNetV2-B6_ML",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-HGNetV2-B6_ML.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "CLIP_vit_base_patch16_448_ML",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "CLIP_vit_base_patch16_448_ML.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x1_0_pedestrian_attribute",
        "suite": "Cls",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "PP-LCNet_x1_0_pedestrian_attribute.yaml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x1_0_vehicle_attribute",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x1_0_vehicle_attribute.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "MobileFaceNet",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "MobileFaceNet.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "ResNet50_face",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "ResNet50_face.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x0_25_textline_ori",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x0_25_textline_ori.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x1_0_textline_ori",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x1_0_textline_ori.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "PP-LCNet_x1_0_table_cls",
        "suite": "Cls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-LCNet_x1_0_table_cls.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)
