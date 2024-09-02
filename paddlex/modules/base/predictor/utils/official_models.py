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

from pathlib import Path

from .....utils.cache import CACHE_DIR
from .....utils.download import download_and_extract

OFFICIAL_MODELS = {
    "ResNet18": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet18_infer.tar",
    "ResNet18_vd": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet18_vd_infer.tar",
    "ResNet34": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet34_infer.tar",
    "ResNet34_vd": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet34_vd_infer.tar",
    "ResNet50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet50_infer.tar",
    "ResNet50_vd": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet50_vd_infer.tar",
    "ResNet101": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet101_infer.tar",
    "ResNet101_vd": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet101_vd_infer.tar",
    "ResNet152": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet152_infer.tar",
    "ResNet152_vd": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet152_vd_infer.tar",
    "ResNet200_vd": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ResNet200_vd_infer.tar",
    "PP-LCNet_x0_25": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x0_25_infer.tar",
    "PP-LCNet_x0_35": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x0_35_infer.tar",
    "PP-LCNet_x0_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x0_5_infer.tar",
    "PP-LCNet_x0_75": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x0_75_infer.tar",
    "PP-LCNet_x1_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x1_0_infer.tar",
    "PP-LCNet_x1_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x1_5_infer.tar",
    "PP-LCNet_x2_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x2_5_infer.tar",
    "PP-LCNet_x2_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNet_x2_0_infer.tar",
    "PP-LCNetV2_small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNetV2_small_infer.tar",
    "PP-LCNetV2_base": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNetV2_base_infer.tar",
    "PP-LCNetV2_large": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LCNetV2_large_infer.tar",
    "MobileNetV3_large_x0_35": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_large_x0_35_infer.tar",
    "MobileNetV3_large_x0_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_large_x0_5_infer.tar",
    "MobileNetV3_large_x0_75": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_large_x0_75_infer.tar",
    "MobileNetV3_large_x1_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_large_x1_0_infer.tar",
    "MobileNetV3_large_x1_25": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_large_x1_25_infer.tar",
    "MobileNetV3_small_x0_35": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_small_x0_35_infer.tar",
    "MobileNetV3_small_x0_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_small_x0_5_infer.tar",
    "MobileNetV3_small_x0_75": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_small_x0_75_infer.tar",
    "MobileNetV3_small_x1_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_small_x1_0_infer.tar",
    "MobileNetV3_small_x1_25": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV3_small_x1_25_infer.tar",
    "ConvNeXt_tiny": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ConvNeXt_tiny_infer.tar",
    "ConvNeXt_small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ConvNeXt_small_infer.tar",
    "ConvNeXt_base_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ConvNeXt_base_224_infer.tar",
    "ConvNeXt_base_384": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ConvNeXt_base_384_infer.tar",
    "ConvNeXt_large_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ConvNeXt_large_224_infer.tar",
    "ConvNeXt_large_384": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/ConvNeXt_large_384_infer.tar",
    "MobileNetV2_x0_25": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV2_x0_25_infer.tar",
    "MobileNetV2_x0_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/MobileNetV2_x0_5_infer.tar",
    "MobileNetV2_x1_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/MobileNetV2_x1_0_infer.tar",
    "MobileNetV2_x1_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/MobileNetV2_x1_5_infer.tar",
    "MobileNetV2_x2_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/MobileNetV2_x2_0_infer.tar",
    "MobileNetV1_x0_25": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV1_x0_25_infer.tar",
    "MobileNetV1_x0_5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV1_x0_5_infer.tar",
    "MobileNetV1_x0_75": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV1_x0_75_infer.tar",
    "MobileNetV1_x1_0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
MobileNetV1_x1_0_infer.tar",
    "SwinTransformer_tiny_patch4_window7_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
SwinTransformer_tiny_patch4_window7_224_infer.tar",
    "SwinTransformer_small_patch4_window7_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
SwinTransformer_small_patch4_window7_224_infer.tar",
    "SwinTransformer_base_patch4_window7_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
SwinTransformer_base_patch4_window7_224_infer.tar",
    "SwinTransformer_base_patch4_window12_384": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
SwinTransformer_base_patch4_window12_384_infer.tar",
    "SwinTransformer_large_patch4_window7_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
SwinTransformer_large_patch4_window7_224_infer.tar",
    "SwinTransformer_large_patch4_window12_384": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
SwinTransformer_large_patch4_window12_384_infer.tar",
    "PP-HGNet_tiny": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNet_tiny_infer.tar",
    "PP-HGNet_small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNet_small_infer.tar",
    "PP-HGNet_base": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNet_base_infer.tar",
    "PP-HGNetV2-B0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNetV2-B0_infer.tar",
    "PP-HGNetV2-B1": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNetV2-B1_infer.tar",
    "PP-HGNetV2-B2": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNetV2-B2_infer.tar",
    "PP-HGNetV2-B3": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNetV2-B3_infer.tar",
    "PP-HGNetV2-B4": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNetV2-B4_infer.tar",
    "PP-HGNetV2-B5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNetV2-B5_infer.tar",
    "PP-HGNetV2-B6": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-HGNetV2-B6_infer.tar",
    "CLIP_vit_base_patch16_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
CLIP_vit_base_patch16_224_infer.tar",
    "CLIP_vit_large_patch14_224": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
CLIP_vit_large_patch14_224_infer.tar",
    "PP-YOLOE_plus-X": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-YOLOE_plus-X_infer.tar",
    "PP-YOLOE_plus-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-YOLOE_plus-L_infer.tar",
    "PP-YOLOE_plus-M": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-YOLOE_plus-M_infer.tar",
    "PP-YOLOE_plus-S": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-YOLOE_plus-S_infer.tar",
    "RT-DETR-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/RT-DETR-L_infer.tar",
    "RT-DETR-H": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/RT-DETR-H_infer.tar",
    "RT-DETR-X": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/RT-DETR-X_infer.tar",
    "YOLOv3-DarkNet53": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/YOLOv3-DarkNet53_infer.tar",
    "YOLOv3-MobileNetV3": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/YOLOv3-MobileNetV3_infer.tar",
    "YOLOv3-ResNet50_vd_DCN": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/YOLOv3-ResNet50_vd_DCN_infer.tar",
    "YOLOX-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/YOLOX-L_infer.tar",
    "YOLOX-M": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/YOLOX-M_infer.tar",
    "YOLOX-N": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/YOLOX-N_infer.tar",
    "YOLOX-S": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/YOLOX-S_infer.tar",
    "YOLOX-T": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/YOLOX-T_infer.tar",
    "YOLOX-X": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/YOLOX-X_infer.tar",
    "RT-DETR-R18": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/RT-DETR-R18_infer.tar",
    "RT-DETR-R50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/RT-DETR-R50_infer.tar",
    "PicoDet-S": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PicoDet-S.tar",
    "PicoDet-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PicoDet-L.tar",
    "Deeplabv3-R50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/Deeplabv3-R50_infer.tar",
    "Deeplabv3-R101": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/Deeplabv3-R101_infer.tar",
    "Deeplabv3_Plus-R50": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
Deeplabv3_Plus-R50_infer.tar",
    "Deeplabv3_Plus-R101": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
Deeplabv3_Plus-R101_infer.tar",
    "PP-LiteSeg-T": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PP-LiteSeg-T_infer.tar",
    "OCRNet_HRNet-W48": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/OCRNet_HRNet-W48_infer.tar",
    "OCRNet_HRNet-W18": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/OCRNet_HRNet-W18_infer.tar",
    "SegFormer-B0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/SegFormer-B0_infer.tar",
    "SegFormer-B1": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/SegFormer-B1_infer.tar",
    "SegFormer-B2": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/SegFormer-B2_infer.tar",
    "SegFormer-B3": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/SegFormer-B3_infer.tar",
    "SegFormer-B4": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/SegFormer-B4_infer.tar",
    "SegFormer-B5": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/SegFormer-B5_infer.tar",
    "SeaFormer_tiny": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/SeaFormer_tiny_infer.tar",
    "SeaFormer_small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/SeaFormer_small_infer.tar",
    "SeaFormer_base": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/SeaFormer_base_infer.tar",
    "SeaFormer_large": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/SeaFormer_large_infer.tar",
    "Mask-RT-DETR-H": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/Mask-RT-DETR-H_infer.tar",
    "Mask-RT-DETR-L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/Mask-RT-DETR-L_infer.tar",
    "PP-OCRv4_server_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
PP-OCRv4_server_rec_infer.tar",
    "PP-OCRv4_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
PP-OCRv4_mobile_rec_infer.tar",
    "PP-OCRv4_server_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
PP-OCRv4_server_det_infer.tar",
    "PP-OCRv4_mobile_det": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
PP-OCRv4_mobile_det_infer.tar",
    "RepSVTR_mobile_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
openatom_rec_repsvtr_ch_infer.tar",
    "SVTRv2_server_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/\
openatom_rec_svtrv2_ch_infer.tar",
    "PicoDet_layout_1x": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/PicoDet-L_layout_infer.tar",
    "SLANet": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/SLANet_infer.tar",
    "LaTeX_OCR_rec": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/LaTeX_OCR_rec_infer.tar",
    "FasterNet_L": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/FasterNet_L_infer.tar",
    "FasterNet_M": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/FasterNet_M_infer.tar",
    "FasterNet_S": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/FasterNet_S_infer.tar",
    "FasterNet_T0": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/FasterNet_T0_infer.tar",
    "FasterNet_T1": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/FasterNet_T1_infer.tar",
    "FasterNet_T2": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/FasterNet_T2_infer.tar",
    "StarNet_S1": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/StarNet_S1_infer.tar",
    "StarNet_S2": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/StarNet_S2_infer.tar",
    "StarNet_S3": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/StarNet_S3_infer.tar",
    "StarNet_S4": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0/StarNet_S4_infer.tar",
}


class OfficialModelsDict(dict):
    """Official Models Dict"""

    def __getitem__(self, key):
        url = super().__getitem__(key)
        save_dir = Path(CACHE_DIR) / "official_models"
        download_and_extract(url, save_dir, f"{key}", overwrite=False)
        return save_dir / f"{key}"


official_models = OfficialModelsDict(OFFICIAL_MODELS)
