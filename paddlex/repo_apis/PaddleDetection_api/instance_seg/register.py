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
from .config import InstanceSegConfig
from .model import InstanceSegModel
from .runner import InstanceSegRunner

REPO_ROOT_PATH = os.environ.get("PADDLE_PDX_PADDLEDETECTION_PATH")
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "configs"))

register_suite_info(
    {
        "suite_name": "InstanceSeg",
        "model": InstanceSegModel,
        "runner": InstanceSegRunner,
        "config": InstanceSegConfig,
        "runner_root_path": REPO_ROOT_PATH,
    }
)

################ Models Using Universal Config ################
register_model_info(
    {
        "model_name": "Mask-RT-DETR-S",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "Mask-RT-DETR-S.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "Mask-RT-DETR-M",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "Mask-RT-DETR-M.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "Mask-RT-DETR-L",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "Mask-RT-DETR-L.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "Mask-RT-DETR-X",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "Mask-RT-DETR-X.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "Mask-RT-DETR-H",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "Mask-RT-DETR-H.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "SOLOv2",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "SOLOv2.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "MaskRCNN-ResNet50",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "MaskRCNN-ResNet50.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "MaskRCNN-ResNet50-FPN",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "MaskRCNN-ResNet50-FPN.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "MaskRCNN-ResNet50-vd-FPN",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "MaskRCNN-ResNet50-vd-FPN.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "MaskRCNN-ResNet101-FPN",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "MaskRCNN-ResNet101-FPN.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "MaskRCNN-ResNet101-vd-FPN",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "MaskRCNN-ResNet101-vd-FPN.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "MaskRCNN-ResNeXt101-vd-FPN",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "MaskRCNN-ResNeXt101-vd-FPN.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "Cascade-MaskRCNN-ResNet50-FPN",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "Cascade-MaskRCNN-ResNet50-FPN.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN",
        "suite": "InstanceSeg",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN.yaml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "PP-YOLOE_seg-S",
        "suite": "InstanceSeg",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE_seg-S.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCOInstSegDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)
