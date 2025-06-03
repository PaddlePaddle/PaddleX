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
from .config import DetConfig
from .model import DetModel
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
        "model_name": "PicoDet_layout_1x_table",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PicoDet_layout_1x_table.yaml"),
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
        "model_name": "FasterRCNN-ResNet34-FPN",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterRCNN-ResNet34-FPN.yaml"),
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
        "model_name": "FasterRCNN-ResNet50",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterRCNN-ResNet50.yaml"),
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
        "model_name": "FasterRCNN-ResNet50-FPN",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterRCNN-ResNet50-FPN.yaml"),
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
        "model_name": "FasterRCNN-ResNet50-vd-FPN",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterRCNN-ResNet50-vd-FPN.yaml"),
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
        "model_name": "FasterRCNN-ResNet50-vd-SSLDv2-FPN",
        "suite": "Det",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml"
        ),
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
        "model_name": "FasterRCNN-ResNet101",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterRCNN-ResNet101.yaml"),
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
        "model_name": "FasterRCNN-ResNet101-FPN",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterRCNN-ResNet101-FPN.yaml"),
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
        "model_name": "FasterRCNN-ResNeXt101-vd-FPN",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterRCNN-ResNeXt101-vd-FPN.yaml"),
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
        "model_name": "FasterRCNN-Swin-Tiny-FPN",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "FasterRCNN-Swin-Tiny-FPN.yaml"),
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
        "model_name": "Cascade-FasterRCNN-ResNet50-FPN",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "Cascade-FasterRCNN-ResNet50-FPN.yaml"),
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
        "model_name": "Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN",
        "suite": "Det",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml"
        ),
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
        "model_name": "PicoDet-XS",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PicoDet-XS.yaml"),
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
        "model_name": "PicoDet-M",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PicoDet-M.yaml"),
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
        "model_name": "FCOS-ResNet50",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "FCOS-ResNet50.yaml"),
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
        "model_name": "DETR-R50",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "DETR-R50.yaml"),
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
        "model_name": "PP-YOLOE-L_vehicle",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE-L_vehicle.yaml"),
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
        "model_name": "PP-YOLOE-S_vehicle",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE-S_vehicle.yaml"),
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
        "model_name": "PP-ShiTuV2_det",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-ShiTuV2_det.yaml"),
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
        "model_name": "PP-YOLOE-L_human",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE-L_human.yaml"),
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
        "model_name": "PP-YOLOE-S_human",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE-S_human.yaml"),
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
        "model_name": "CenterNet-DLA-34",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "CenterNet-DLA-34.yaml"),
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
        "model_name": "CenterNet-ResNet50",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "CenterNet-ResNet50.yaml"),
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
        "model_name": "PP-YOLOE_plus_SOD-L",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE_plus_SOD-L.yaml"),
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
        "model_name": "PP-YOLOE_plus_SOD-S",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE_plus_SOD-S.yaml"),
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
        "model_name": "PP-YOLOE_plus_SOD-largesize-L",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE_plus_SOD-largesize-L.yaml"),
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
        "model_name": "RT-DETR-H_layout_3cls",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "RT-DETR-H_layout_3cls.yaml"),
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
        "model_name": "PicoDet-S_layout_3cls",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PicoDet-S_layout_3cls.yaml"),
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
        "model_name": "PicoDet-S_layout_17cls",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PicoDet-S_layout_17cls.yaml"),
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
        "model_name": "PicoDet-L_layout_3cls",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PicoDet-L_layout_3cls.yaml"),
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
        "model_name": "PicoDet-L_layout_17cls",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PicoDet-L_layout_17cls.yaml"),
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
        "model_name": "RT-DETR-H_layout_17cls",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "RT-DETR-H_layout_17cls.yaml"),
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

register_model_info(
    {
        "model_name": "BlazeFace",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "BlazeFace.yaml"),
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
        "model_name": "BlazeFace-FPN-SSH",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "BlazeFace-FPN-SSH.yaml"),
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
        "model_name": "PP-YOLOE_plus-S_face",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE_plus-S_face.yaml"),
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
        "model_name": "PP-YOLOE-R-L",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-YOLOE-R-L.yaml"),
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
        "model_name": "RT-DETR-L_wired_table_cell_det",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "RT-DETR-L_wired_table_cell_det.yaml"),
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
        "model_name": "RT-DETR-L_wireless_table_cell_det",
        "suite": "Det",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "RT-DETR-L_wireless_table_cell_det.yaml"
        ),
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
        "model_name": "Co-Deformable-DETR-R50",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "Co-Deformable-DETR-R50.yaml"),
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
        "model_name": "Co-Deformable-DETR-Swin-T",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "Co-Deformable-DETR-Swin-T.yaml"),
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
        "model_name": "Co-DINO-R50",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "Co-DINO-R50.yaml"),
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
        "model_name": "Co-DINO-Swin-L",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "Co-DINO-Swin-L.yaml"),
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
        "model_name": "PP-TinyPose_128x96",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-TinyPose_128x96.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["KeypointTopDownCocoDetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "PP-TinyPose_256x192",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-TinyPose_256x192.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["KeypointTopDownCocoDetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)

register_model_info(
    {
        "model_name": "PP-DocLayout-L",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-DocLayout-L.yaml"),
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
        "model_name": "PP-DocLayout-M",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-DocLayout-M.yaml"),
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
        "model_name": "PP-DocLayout-S",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-DocLayout-S.yaml"),
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
        "model_name": "PP-DocLayout_plus-L",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-DocLayout_plus-L.yaml"),
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
        "model_name": "PP-DocBlockLayout",
        "suite": "Det",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-DocBlockLayout.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "supported_dataset_types": ["COCODetDataset"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": ["OFF"],
        },
    }
)
