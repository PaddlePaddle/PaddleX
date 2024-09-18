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
import os.path as osp

from ...base.register import register_model_info, register_suite_info
from .model import DetModel
from .config import DetConfig
from .runner import DetRunner

REPO_ROOT_PATH = os.environ.get("PADDLE_PDX_PADDLEDETECTION_PATH")
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "configs"))

register_suite_info(
    {
        "suite_name": "Det",
        "model": DetModel,
        "runner": DetRunner,
        "config": DetConfig,
        "runner_root_path": REPO_ROOT_PATH,
    }
)

################ Models Using Universal Config ################
register_model_info(
    {
        "model_name": "PicoDet-S",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PicoDet-S.yaml"),
        "auto_compression_config_path": osp.join(
            PDX_CONFIG_DIR, "slim", "picodet_s_lcnet_qat.yml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export", "compression"],
        "supported_dataset_types": ["COCODetDataset"],
    }
)

register_model_info(
    {
        "model_name": "PicoDet-L",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PicoDet-L.yaml"),
        "auto_compression_config_path": osp.join(
            PDX_CONFIG_DIR, "slim", "picodet_l_lcnet_qat.yml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export", "compression"],
        "supported_dataset_types": ["COCODetDataset"],
    }
)

register_model_info(
    {
        "model_name": "PP-YOLOE_plus-S",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE_plus-S.yaml"),
        "auto_compression_config_path": osp.join(
            PDX_CONFIG_DIR, "slim", "ppyoloe_plus_crn_s_qat.yml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export", "compression"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx"],
            "dy2st": False,
            "amp": ["O1", "O2"],
        },
    }
)

register_model_info(
    {
        "model_name": "PP-YOLOE_plus-M",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE_plus-M.yaml"),
        "auto_compression_config_path": osp.join(
            PDX_CONFIG_DIR, "slim", "ppyoloe_plus_crn_l_qat.yml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export", "compression"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx"],
            "dy2st": False,
            "amp": ["O1", "O2"],
        },
    }
)

register_model_info(
    {
        "model_name": "PP-YOLOE_plus-L",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE_plus-L.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["O1", "O2"],
        },
    }
)

register_model_info(
    {
        "model_name": "PP-YOLOE_plus-X",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE_plus-X.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["O1", "O2"],
        },
    }
)

register_model_info(
    {
        "model_name": "RT-DETR-L",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "RT-DETR-L.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "RT-DETR-H",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "RT-DETR-H.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "RT-DETR-X",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "RT-DETR-X.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "RT-DETR-R18",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "RT-DETR-R18.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "RT-DETR-R50",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "RT-DETR-R50.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "PicoDet_layout_1x",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PicoDet_layout_1x.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "YOLOv3-DarkNet53",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "YOLOv3-DarkNet53.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "YOLOv3-MobileNetV3",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "YOLOv3-MobileNetV3.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "YOLOv3-ResNet50_vd_DCN",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "YOLOv3-ResNet50_vd_DCN.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "YOLOX-L",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "YOLOX-L.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "YOLOX-M",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "YOLOX-M.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "YOLOX-N",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "YOLOX-N.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "YOLOX-S",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "YOLOX-S.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "YOLOX-T",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "YOLOX-T.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "YOLOX-X",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "YOLOX-X.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "PicoDet_LCNet_x2_5_face",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PicoDet_LCNet_x2_5_face.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)