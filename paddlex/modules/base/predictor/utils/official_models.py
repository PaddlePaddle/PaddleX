# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: PaddlePaddle Authors
"""
from pathlib import Path

from .....utils.cache import CACHE_DIR
from .....utils.download import download_and_extract

OFFICIAL_MODELS = {
    "ResNet18":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet18_infer.tar",
    "ResNet34":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet34_infer.tar",
    "ResNet50":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet50_infer.tar",
    "ResNet101":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet101_infer.tar",
    "ResNet152":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet152_infer.tar",
    "PP-LCNet_x0_25":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x0_25_infer.tar",
    "PP-LCNet_x0_35":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x0_35_infer.tar",
    "PP-LCNet_x0_5":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x0_5_infer.tar",
    "PP-LCNet_x0_75":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x0_75_infer.tar",
    "PP-LCNet_x1_0":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x1_0_infer.tar",
    "PP-LCNet_x1_5":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x1_5_infer.tar",
    "PP-LCNet_x2_5":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x2_5_infer.tar",
    "PP-LCNet_x2_0":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x2_0_infer.tar",
    "MobileNetV3_large_x0_35":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_large_x0_35_infer.tar",
    "MobileNetV3_large_x0_5":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_large_x0_5_infer.tar",
    "MobileNetV3_large_x0_75":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_large_x0_75_infer.tar",
    "MobileNetV3_large_x1_0":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_large_x1_0_infer.tar",
    "MobileNetV3_large_x1_25":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_large_x1_25_infer.tar",
    "MobileNetV3_small_x0_35":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_small_x0_35_infer.tar",
    "MobileNetV3_small_x0_5":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_small_x0_5_infer.tar",
    "MobileNetV3_small_x0_75":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_small_x0_75_infer.tar",
    "MobileNetV3_small_x1_0":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_small_x1_0_infer.tar",
    "MobileNetV3_small_x1_25":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_small_x1_25_infer.tar",
    "ConvNeXt_tiny":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ConvNeXt_tiny_infer.tar",
    "MobileNetV2_x0_25":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV2_x0_25_infer.tar",
    "MobileNetV2_x0_5":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/MobileNetV2_x0_5_infer.tar",
    "MobileNetV2_x1_0":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/MobileNetV2_x1_0_infer.tar",
    "MobileNetV2_x1_5":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/MobileNetV2_x1_5_infer.tar",
    "MobileNetV2_x2_0":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/MobileNetV2_x2_0_infer.tar",
    "SwinTransformer_base_patch4_window7_224":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
SwinTransformer_base_patch4_window7_224_infer.tar",
    "PP-HGNet_small":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNet_small_infer.tar",
    "PP-HGNetV2-B0":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNetV2-B0_infer.tar",
    "PP-HGNetV2-B4":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNetV2-B4_infer.tar",
    "PP-HGNetV2-B6":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNetV2-B6_infer.tar",
    "CLIP_vit_base_patch16_224":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
CLIP_vit_base_patch16_224_infer.tar",
    "CLIP_vit_large_patch14_224":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
CLIP_vit_large_patch14_224_infer.tar",
    "PP-YOLOE_plus-X":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-YOLOE_plus-X_infer.tar",
    "PP-YOLOE_plus-L":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-YOLOE_plus-L_infer.tar",
    "PP-YOLOE_plus-M":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-YOLOE_plus-M_infer.tar",
    "PP-YOLOE_plus-S":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-YOLOE_plus-S_infer.tar",
    "RT-DETR-L":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/RT-DETR-L_infer.tar",
    "RT-DETR-H":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/RT-DETR-H_infer.tar",
    "RT-DETR-X":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/RT-DETR-X_infer.tar",
    "RT-DETR-R18":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/RT-DETR-R18_infer.tar",
    "RT-DETR-R50":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/RT-DETR-R50_infer.tar",
    "PicoDet-S":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PicoDet-S.tar",
    "PicoDet-L":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PicoDet-L.tar",
    "Deeplabv3-R50":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/Deeplabv3-R50_infer.tar",
    "Deeplabv3-R101":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/Deeplabv3-R101_infer.tar",
    "Deeplabv3_Plus-R50":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
Deeplabv3_Plus-R50_infer.tar",
    "Deeplabv3_Plus-R101":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
Deeplabv3_Plus-R101_infer.tar",
    "PP-LiteSeg-T":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LiteSeg-T_infer.tar",
    "OCRNet_HRNet-W48":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/OCRNet_HRNet-W48_infer.tar",
    "Mask-RT-DETR-H":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/Mask-RT-DETR-H_infer.tar",
    "Mask-RT-DETR-L":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/Mask-RT-DETR-L_infer.tar",
    "PP-OCRv4_server_rec":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
PP-OCRv4_server_rec_infer.tar",
    "PP-OCRv4_mobile_rec":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
PP-OCRv4_mobile_rec_infer.tar",
    "PP-OCRv4_server_det":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
PP-OCRv4_server_det_infer.tar",
    "PP-OCRv4_mobile_det":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
PP-OCRv4_mobile_det_infer.tar",
    "PicoDet_layout_1x":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PicoDet-L_layout_infer.tar",
    "SLANet":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/SLANet_infer.tar",
}


class OfficialModelsDict(dict):
    """Official Models Dict
    """

    def __getitem__(self, key):
        url = super().__getitem__(key)
        save_dir = Path(CACHE_DIR) / "official_models"
        download_and_extract(url, save_dir, f"{key}", overwrite=False)
        return save_dir / f"{key}"


official_models = OfficialModelsDict(OFFICIAL_MODELS)
