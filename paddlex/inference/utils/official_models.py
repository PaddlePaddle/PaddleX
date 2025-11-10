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
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import huggingface_hub as hf_hub

hf_hub.logging.set_verbosity_error()

import modelscope
import requests

os.environ["AISTUDIO_LOG"] = "critical"
from aistudio_sdk.snapshot_download import snapshot_download as aistudio_download

from ...utils import logging
from ...utils.cache import CACHE_DIR
from ...utils.download import download_and_extract
from ...utils.flags import MODEL_SOURCE

ALL_MODELS = [
    "ResNet18",
    "ResNet18_vd",
    "ResNet34",
    "ResNet34_vd",
    "ResNet50",
    "ResNet50_vd",
    "ResNet101",
    "ResNet101_vd",
    "ResNet152",
    "ResNet152_vd",
    "ResNet200_vd",
    "PaddleOCR-VL",
    "PP-LCNet_x0_25",
    "PP-LCNet_x0_25_textline_ori",
    "PP-LCNet_x0_35",
    "PP-LCNet_x0_5",
    "PP-LCNet_x0_75",
    "PP-LCNet_x1_0",
    "PP-LCNet_x1_0_doc_ori",
    "PP-LCNet_x1_0_textline_ori",
    "PP-LCNet_x1_5",
    "PP-LCNet_x2_5",
    "PP-LCNet_x2_0",
    "PP-LCNetV2_small",
    "PP-LCNetV2_base",
    "PP-LCNetV2_large",
    "MobileNetV3_large_x0_35",
    "MobileNetV3_large_x0_5",
    "MobileNetV3_large_x0_75",
    "MobileNetV3_large_x1_0",
    "MobileNetV3_large_x1_25",
    "MobileNetV3_small_x0_35",
    "MobileNetV3_small_x0_5",
    "MobileNetV3_small_x0_75",
    "MobileNetV3_small_x1_0",
    "MobileNetV3_small_x1_25",
    "ConvNeXt_tiny",
    "ConvNeXt_small",
    "ConvNeXt_base_224",
    "ConvNeXt_base_384",
    "ConvNeXt_large_224",
    "ConvNeXt_large_384",
    "MobileNetV2_x0_25",
    "MobileNetV2_x0_5",
    "MobileNetV2_x1_0",
    "MobileNetV2_x1_5",
    "MobileNetV2_x2_0",
    "MobileNetV1_x0_25",
    "MobileNetV1_x0_5",
    "MobileNetV1_x0_75",
    "MobileNetV1_x1_0",
    "SwinTransformer_tiny_patch4_window7_224",
    "SwinTransformer_small_patch4_window7_224",
    "SwinTransformer_base_patch4_window7_224",
    "SwinTransformer_base_patch4_window12_384",
    "SwinTransformer_large_patch4_window7_224",
    "SwinTransformer_large_patch4_window12_384",
    "PP-HGNet_tiny",
    "PP-HGNet_small",
    "PP-HGNet_base",
    "PP-HGNetV2-B0",
    "PP-HGNetV2-B1",
    "PP-HGNetV2-B2",
    "PP-HGNetV2-B3",
    "PP-HGNetV2-B4",
    "PP-HGNetV2-B5",
    "PP-HGNetV2-B6",
    "FasterNet-L",
    "FasterNet-M",
    "FasterNet-S",
    "FasterNet-T0",
    "FasterNet-T1",
    "FasterNet-T2",
    "StarNet-S1",
    "StarNet-S2",
    "StarNet-S3",
    "StarNet-S4",
    "MobileNetV4_conv_small",
    "MobileNetV4_conv_medium",
    "MobileNetV4_conv_large",
    "MobileNetV4_hybrid_medium",
    "MobileNetV4_hybrid_large",
    "CLIP_vit_base_patch16_224",
    "CLIP_vit_large_patch14_224",
    "PP-LCNet_x1_0_ML",
    "PP-HGNetV2-B0_ML",
    "PP-HGNetV2-B4_ML",
    "PP-HGNetV2-B6_ML",
    "ResNet50_ML",
    "CLIP_vit_base_patch16_448_ML",
    "PP-YOLOE_plus-X",
    "PP-YOLOE_plus-L",
    "PP-YOLOE_plus-M",
    "PP-YOLOE_plus-S",
    "RT-DETR-L",
    "RT-DETR-H",
    "RT-DETR-X",
    "YOLOv3-DarkNet53",
    "YOLOv3-MobileNetV3",
    "YOLOv3-ResNet50_vd_DCN",
    "YOLOX-L",
    "YOLOX-M",
    "YOLOX-N",
    "YOLOX-S",
    "YOLOX-T",
    "YOLOX-X",
    "RT-DETR-R18",
    "RT-DETR-R50",
    "PicoDet-S",
    "PicoDet-L",
    "Deeplabv3-R50",
    "Deeplabv3-R101",
    "Deeplabv3_Plus-R50",
    "Deeplabv3_Plus-R101",
    "PP-ShiTuV2_rec",
    "PP-ShiTuV2_rec_CLIP_vit_base",
    "PP-ShiTuV2_rec_CLIP_vit_large",
    "PP-LiteSeg-T",
    "PP-LiteSeg-B",
    "OCRNet_HRNet-W48",
    "OCRNet_HRNet-W18",
    "SegFormer-B0",
    "SegFormer-B1",
    "SegFormer-B2",
    "SegFormer-B3",
    "SegFormer-B4",
    "SegFormer-B5",
    "SeaFormer_tiny",
    "SeaFormer_small",
    "SeaFormer_base",
    "SeaFormer_large",
    "Mask-RT-DETR-H",
    "Mask-RT-DETR-L",
    "PP-OCRv4_server_rec",
    "Mask-RT-DETR-S",
    "Mask-RT-DETR-M",
    "Mask-RT-DETR-X",
    "SOLOv2",
    "MaskRCNN-ResNet50",
    "MaskRCNN-ResNet50-FPN",
    "MaskRCNN-ResNet50-vd-FPN",
    "MaskRCNN-ResNet101-FPN",
    "MaskRCNN-ResNet101-vd-FPN",
    "MaskRCNN-ResNeXt101-vd-FPN",
    "Cascade-MaskRCNN-ResNet50-FPN",
    "Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN",
    "PP-YOLOE_seg-S",
    "PP-OCRv3_mobile_rec",
    "en_PP-OCRv3_mobile_rec",
    "korean_PP-OCRv3_mobile_rec",
    "japan_PP-OCRv3_mobile_rec",
    "chinese_cht_PP-OCRv3_mobile_rec",
    "te_PP-OCRv3_mobile_rec",
    "ka_PP-OCRv3_mobile_rec",
    "ta_PP-OCRv3_mobile_rec",
    "latin_PP-OCRv3_mobile_rec",
    "arabic_PP-OCRv3_mobile_rec",
    "cyrillic_PP-OCRv3_mobile_rec",
    "devanagari_PP-OCRv3_mobile_rec",
    "en_PP-OCRv4_mobile_rec",
    "PP-OCRv4_server_rec_doc",
    "PP-OCRv4_mobile_rec",
    "PP-OCRv4_server_det",
    "PP-OCRv4_mobile_det",
    "PP-OCRv3_server_det",
    "PP-OCRv3_mobile_det",
    "PP-OCRv4_server_seal_det",
    "PP-OCRv4_mobile_seal_det",
    "ch_RepSVTR_rec",
    "ch_SVTRv2_rec",
    "PP-LCNet_x1_0_pedestrian_attribute",
    "PP-LCNet_x1_0_vehicle_attribute",
    "PicoDet_layout_1x",
    "PicoDet_layout_1x_table",
    "SLANet",
    "SLANet_plus",
    "LaTeX_OCR_rec",
    "UniMERNet",
    "PP-FormulaNet-S",
    "PP-FormulaNet-L",
    "PP-FormulaNet_plus-S",
    "PP-FormulaNet_plus-M",
    "PP-FormulaNet_plus-L",
    "FasterRCNN-ResNet34-FPN",
    "FasterRCNN-ResNet50",
    "FasterRCNN-ResNet50-FPN",
    "FasterRCNN-ResNet50-vd-FPN",
    "FasterRCNN-ResNet50-vd-SSLDv2-FPN",
    "FasterRCNN-ResNet101",
    "FasterRCNN-ResNet101-FPN",
    "FasterRCNN-ResNeXt101-vd-FPN",
    "FasterRCNN-Swin-Tiny-FPN",
    "Cascade-FasterRCNN-ResNet50-FPN",
    "Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN",
    "UVDoc",
    "DLinear",
    "NLinear",
    "RLinear",
    "Nonstationary",
    "TimesNet",
    "TiDE",
    "PatchTST",
    "DLinear_ad",
    "AutoEncoder_ad",
    "Nonstationary_ad",
    "PatchTST_ad",
    "TimesNet_ad",
    "TimesNet_cls",
    "STFPM",
    "FCOS-ResNet50",
    "DETR-R50",
    "PP-YOLOE-L_vehicle",
    "PP-YOLOE-S_vehicle",
    "PP-ShiTuV2_det",
    "PP-YOLOE-S_human",
    "PP-YOLOE-L_human",
    "PicoDet-M",
    "PicoDet-XS",
    "PP-YOLOE_plus_SOD-L",
    "PP-YOLOE_plus_SOD-S",
    "PP-YOLOE_plus_SOD-largesize-L",
    "CenterNet-DLA-34",
    "CenterNet-ResNet50",
    "PicoDet-S_layout_3cls",
    "PicoDet-S_layout_17cls",
    "PicoDet-L_layout_3cls",
    "PicoDet-L_layout_17cls",
    "RT-DETR-H_layout_3cls",
    "RT-DETR-H_layout_17cls",
    "PicoDet_LCNet_x2_5_face",
    "BlazeFace",
    "BlazeFace-FPN-SSH",
    "PP-YOLOE_plus-S_face",
    "MobileFaceNet",
    "ResNet50_face",
    "PP-YOLOE-R-L",
    "Co-Deformable-DETR-R50",
    "Co-Deformable-DETR-Swin-T",
    "Co-DINO-R50",
    "Co-DINO-Swin-L",
    "whisper_large",
    "whisper_base",
    "whisper_medium",
    "whisper_small",
    "whisper_tiny",
    "PP-TSM-R50_8frames_uniform",
    "PP-TSMv2-LCNetV2_8frames_uniform",
    "PP-TSMv2-LCNetV2_16frames_uniform",
    "MaskFormer_tiny",
    "MaskFormer_small",
    "PP-LCNet_x1_0_table_cls",
    "SLANeXt_wired",
    "SLANeXt_wireless",
    "RT-DETR-L_wired_table_cell_det",
    "RT-DETR-L_wireless_table_cell_det",
    "YOWO",
    "PP-TinyPose_128x96",
    "PP-TinyPose_256x192",
    "GroundingDINO-T",
    "SAM-H_box",
    "SAM-H_point",
    "PP-DocLayoutV2",
    "PP-DocLayout-L",
    "PP-DocLayout-M",
    "PP-DocLayout-S",
    "PP-DocLayout_plus-L",
    "PP-DocBlockLayout",
    "BEVFusion",
    "YOLO-Worldv2-L",
    "PP-DocBee-2B",
    "PP-DocBee-7B",
    "PP-Chart2Table",
    "PP-OCRv5_server_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv5_server_rec",
    "PP-OCRv5_mobile_rec",
    "eslav_PP-OCRv5_mobile_rec",
    "PP-DocBee2-3B",
    "latin_PP-OCRv5_mobile_rec",
    "korean_PP-OCRv5_mobile_rec",
    "th_PP-OCRv5_mobile_rec",
    "el_PP-OCRv5_mobile_rec",
    "en_PP-OCRv5_mobile_rec",
    "arabic_PP-OCRv5_mobile_rec",
    "te_PP-OCRv5_mobile_rec",
    "ta_PP-OCRv5_mobile_rec",
    "devanagari_PP-OCRv5_mobile_rec",
    "cyrillic_PP-OCRv5_mobile_rec",
]


OCR_MODELS = [
    "arabic_PP-OCRv3_mobile_rec",
    "chinese_cht_PP-OCRv3_mobile_rec",
    "ch_RepSVTR_rec",
    "ch_SVTRv2_rec",
    "cyrillic_PP-OCRv3_mobile_rec",
    "devanagari_PP-OCRv3_mobile_rec",
    "en_PP-OCRv3_mobile_rec",
    "en_PP-OCRv4_mobile_rec",
    "eslav_PP-OCRv5_mobile_rec",
    "japan_PP-OCRv3_mobile_rec",
    "ka_PP-OCRv3_mobile_rec",
    "korean_PP-OCRv3_mobile_rec",
    "korean_PP-OCRv5_mobile_rec",
    "LaTeX_OCR_rec",
    "latin_PP-OCRv3_mobile_rec",
    "latin_PP-OCRv5_mobile_rec",
    "en_PP-OCRv5_mobile_rec",
    "th_PP-OCRv5_mobile_rec",
    "el_PP-OCRv5_mobile_rec",
    "PaddleOCR-VL",
    "PicoDet_layout_1x",
    "PicoDet_layout_1x_table",
    "PicoDet-L_layout_17cls",
    "PicoDet-L_layout_3cls",
    "PicoDet-S_layout_17cls",
    "PicoDet-S_layout_3cls",
    "PP-DocBee2-3B",
    "PP-Chart2Table",
    "PP-DocBee-2B",
    "PP-DocBee-7B",
    "PP-DocBlockLayout",
    "PP-DocLayoutV2",
    "PP-DocLayout-L",
    "PP-DocLayout-M",
    "PP-DocLayout_plus-L",
    "PP-DocLayout-S",
    "PP-DocLayoutV2",
    "PP-FormulaNet-L",
    "PP-FormulaNet_plus-L",
    "PP-FormulaNet_plus-M",
    "PP-FormulaNet_plus-S",
    "PP-FormulaNet-S",
    "PP-LCNet_x0_25_textline_ori",
    "PP-LCNet_x1_0_doc_ori",
    "PP-LCNet_x1_0_table_cls",
    "PP-LCNet_x1_0_textline_ori",
    "PP-OCRv3_mobile_det",
    "PP-OCRv3_mobile_rec",
    "PP-OCRv3_server_det",
    "PP-OCRv4_mobile_det",
    "PP-OCRv4_mobile_rec",
    "PP-OCRv4_mobile_seal_det",
    "PP-OCRv4_server_det",
    "PP-OCRv4_server_rec_doc",
    "PP-OCRv4_server_rec",
    "PP-OCRv4_server_seal_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv5_mobile_rec",
    "PP-OCRv5_server_det",
    "PP-OCRv5_server_rec",
    "RT-DETR-H_layout_17cls",
    "RT-DETR-H_layout_3cls",
    "RT-DETR-L_wired_table_cell_det",
    "RT-DETR-L_wireless_table_cell_det",
    "SLANet",
    "SLANet_plus",
    "SLANeXt_wired",
    "SLANeXt_wireless",
    "ta_PP-OCRv3_mobile_rec",
    "te_PP-OCRv3_mobile_rec",
    "UniMERNet",
    "UVDoc",
    "arabic_PP-OCRv5_mobile_rec",
    "te_PP-OCRv5_mobile_rec",
    "ta_PP-OCRv5_mobile_rec",
    "devanagari_PP-OCRv5_mobile_rec",
    "cyrillic_PP-OCRv5_mobile_rec",
]


class _BaseModelHoster(ABC):
    alias = ""
    model_list = []
    healthcheck_url = None
    _healthcheck_timeout = 1

    def __init__(self, save_dir):
        self._save_dir = save_dir

    def get_model(self, model_name):
        assert (
            model_name in self.model_list
        ), f"The model {model_name} is not supported on hosting {self.__class__.__name__}!"

        model_dir = self._save_dir / f"{model_name}"
        logging.info(
            f"Using official model ({model_name}), the model files will be automatically downloaded and saved in `{model_dir}`."
        )
        self._download(model_name, model_dir)
        logging.debug(
            f"`{model_name}` model files has been download from model source: `{self.alias}`!"
        )

        return model_dir

    @abstractmethod
    def _download(self):
        raise NotImplementedError

    @classmethod
    def is_available(cls):
        if cls.healthcheck_url is None:
            return True
        try:
            response = requests.head(
                cls.healthcheck_url, timeout=cls._healthcheck_timeout
            )
            return response.ok == True
        except Exception:
            logging.debug(f"The model hosting platform({cls.__name__}) is unreachable!")
            return False


class _BosModelHoster(_BaseModelHoster):
    model_list = ALL_MODELS
    alias = "bos"
    healthcheck_url = "https://paddle-model-ecology.bj.bcebos.com"

    version = "paddle3.0.0"
    base_url = (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model"
    )
    special_model_fn = {
        "whisper_large": "whisper_large.tar",
        "whisper_base": "whisper_base.tar",
        "whisper_medium": "whisper_medium.tar",
        "whisper_small": "whisper_small.tar",
        "whisper_tiny": "whisper_tiny.tar",
    }

    def _download(self, model_name, save_dir):
        if model_name in self.special_model_fn:
            fn = self.special_model_fn[model_name]
        else:
            fn = f"{model_name}_infer.tar"
        url = f"{self.base_url}/{self.version}/{fn}"
        download_and_extract(url, save_dir.parent, model_name, overwrite=False)


class _HuggingFaceModelHoster(_BaseModelHoster):
    model_list = OCR_MODELS
    alias = "huggingface"
    healthcheck_url = "https://huggingface.co"

    def _download(self, model_name, save_dir):
        def _clone(local_dir):
            hf_hub.snapshot_download(
                repo_id=f"PaddlePaddle/{model_name}", local_dir=local_dir
            )

        if os.path.exists(save_dir):
            _clone(save_dir)
        else:
            with tempfile.TemporaryDirectory() as td:
                temp_dir = os.path.join(td, "temp_dir")
                _clone(temp_dir)
                shutil.move(temp_dir, save_dir)


class _ModelScopeModelHoster(_BaseModelHoster):
    model_list = OCR_MODELS
    alias = "modelscope"
    healthcheck_url = "https://modelscope.cn"

    def _download(self, model_name, save_dir):
        def _clone(local_dir):
            modelscope.snapshot_download(
                repo_id=f"PaddlePaddle/{model_name}", local_dir=local_dir
            )

        if os.path.exists(save_dir):
            _clone(save_dir)
        else:
            with tempfile.TemporaryDirectory() as td:
                temp_dir = os.path.join(td, "temp_dir")
                _clone(temp_dir)
                shutil.move(temp_dir, save_dir)


class _AIStudioModelHoster(_BaseModelHoster):
    model_list = OCR_MODELS
    alias = "aistudio"
    healthcheck_url = "https://aistudio.baidu.com"

    def _download(self, model_name, save_dir):
        def _clone(local_dir):
            if model_name == "PaddleOCR-VL":
                aistudio_download(
                    repo_id=f"PaddlePaddle/{model_name}", local_dir=local_dir
                )
            else:
                aistudio_download(repo_id=f"PaddleX/{model_name}", local_dir=local_dir)

        if os.path.exists(save_dir):
            _clone(save_dir)
        else:
            with tempfile.TemporaryDirectory() as td:
                temp_dir = os.path.join(td, "temp_dir")
                _clone(temp_dir)
                shutil.move(temp_dir, save_dir)


class _ModelManager:
    model_list = ALL_MODELS
    _save_dir = Path(CACHE_DIR) / "official_models"

    def __init__(self) -> None:
        self._hosters = self._build_hosters()

    def _build_hosters(self):
        hosters = []
        for hoster_cls in [
            _HuggingFaceModelHoster,
            _AIStudioModelHoster,
            _ModelScopeModelHoster,
            _BosModelHoster,
        ]:
            if hoster_cls.alias == MODEL_SOURCE:
                if hoster_cls.is_available():
                    hosters.insert(0, hoster_cls(self._save_dir))
            else:
                if hoster_cls.is_available():
                    hosters.append(hoster_cls(self._save_dir))
        if len(hosters) == 0:
            logging.warning(
                f"No model hoster is available! Please check your network connection to one of the following model hosts: HuggingFace ({_HuggingFaceModelHoster.healthcheck_url}), ModelScope ({_ModelScopeModelHoster.healthcheck_url}), AIStudio ({_AIStudioModelHoster.healthcheck_url}), or BOS ({_BosModelHoster.healthcheck_url}). Otherwise, only local models can be used."
            )
        return hosters

    def _get_model_local_path(self, model_name):
        if model_name == "PaddleOCR-VL-0.9B":
            model_name = "PaddleOCR-VL"

        model_dir = self._save_dir / f"{model_name}"
        if os.path.exists(model_dir):
            logging.info(
                f"Model files already exist. Using cached files. To redownload, please delete the directory manually: `{model_dir}`."
            )
        else:
            if len(self._hosters) == 0:
                msg = "No available model hosting platforms detected. Please check your network connection."
                logging.error(msg)
                raise Exception(msg)

            model_dir = self._download_from_hoster(self._hosters, model_name)

        if model_name == "PaddleOCR-VL":
            vl_model_dir = model_dir / "PaddleOCR-VL-0.9B"
            if vl_model_dir.exists() and vl_model_dir.is_dir():
                return vl_model_dir

        return model_dir

    def _download_from_hoster(self, hosters, model_name):
        for idx, hoster in enumerate(hosters):
            if model_name in hoster.model_list:
                try:
                    model_path = hoster.get_model(model_name)
                    return model_path

                except Exception as e:
                    if len(hosters) <= 1:
                        raise Exception(
                            f"Encounter exception when download model from {hoster.alias}. No model source is available! Please check network or use local model files!"
                        )
                    logging.warning(
                        f"Encountering exception when download model from {hoster.alias}: \n{e}, will try to download from other model sources: `{hosters[idx + 1].alias}`."
                    )
                    return self._download_from_hoster(hosters[idx + 1 :], model_name)
        raise Exception(
            f"No model source is available for model `{model_name}`! Please check model name and network, or use local model files!"
        )

    def __contains__(self, model_name):
        return model_name in self.model_list

    def __getitem__(self, model_name):
        return self._get_model_local_path(model_name)


official_models = _ModelManager()
