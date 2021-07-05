# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import glob
import paddle
import paddlex.utils.logging as logging
from .download import download_and_decompress

seg_pretrain_weights_dict = {
    'UNet': ['CITYSCAPES'],
    'DeepLabV3P': ['CITYSCAPES', 'PascalVOC', 'IMAGENET'],
    'FastSCNN': ['CITYSCAPES'],
    'HRNet': ['CITYSCAPES', 'PascalVOC'],
    'BiSeNetV2': ['CITYSCAPES']
}

det_pretrain_weights_dict = {
    'YOLOv3_MobileNetV1': ['COCO', 'PascalVOC', 'IMAGENET'],
    'YOLOv3_MobileNetV1_ssld': ['COCO', 'PascalVOC', 'IMAGENET'],
    'YOLOv3_DarkNet53': ['COCO', 'IMAGENET'],
    'YOLOv3_ResNet50_vd_dcn': ['COCO', 'IMAGENET'],
    'YOLOv3_ResNet34': ['COCO', 'IMAGENET'],
    'YOLOv3_MobileNetV3': ['COCO', 'PascalVOC', 'IMAGENET'],
    'YOLOv3_MobileNetV3_ssld': ['PascalVOC', 'IMAGENET'],
    'FasterRCNN_ResNet50_vd': ['COCO', 'IMAGENET'],
    'FasterRCNN_ResNet50_vd_fpn': ['COCO', 'IMAGENET'],
    'FasterRCNN_ResNet50': ['COCO', 'IMAGENET'],
    'FasterRCNN_ResNet50_fpn': ['COCO', 'IMAGENET'],
    'FasterRCNN_ResNet34_fpn': ['COCO', 'IMAGENET'],
    'FasterRCNN_ResNet34_vd_fpn': ['COCO', 'IMAGENET'],
    'FasterRCNN_ResNet101_fpn': ['COCO', 'IMAGENET'],
    'FasterRCNN_ResNet101_vd_fpn': ['COCO', 'IMAGENET'],
    'FasterRCNN_ResNet50_vd_ssld_fpn': ['COCO', 'IMAGENET'],
    'FasterRCNN_HRNet_W18_fpn': ['COCO', 'IMAGENET'],
    'PPYOLO_ResNet50_vd_dcn': ['COCO', 'IMAGENET'],
    'PPYOLO_ResNet18_vd': ['COCO', 'IMAGENET'],
    'PPYOLO_MobileNetV3_large': ['COCO', 'IMAGENET'],
    'PPYOLO_MobileNetV3_small': ['COCO', 'IMAGENET'],
    'PPYOLOv2_ResNet50_vd_dcn': ['COCO', 'IMAGENET'],
    'PPYOLOv2_ResNet101_vd_dcn': ['COCO', 'IMAGENET'],
    'PPYOLOTiny_MobileNetV3': ['COCO', 'IMAGENET'],
    'MaskRCNN_ResNet50': ['COCO', 'IMAGENET'],
    'MaskRCNN_ResNet50_fpn': ['COCO', 'IMAGENET'],
    'MaskRCNN_ResNet50_vd_fpn': ['COCO', 'IMAGENET'],
    'MaskRCNN_ResNet50_vd_ssld_fpn': ['COCO', 'IMAGENET'],
    'MaskRCNN_ResNet101_fpn': ['COCO', 'IMAGENET'],
    'MaskRCNN_ResNet101_vd_fpn': ['COCO', 'IMAGENET']
}

cityscapes_weights = {
    'UNet_CITYSCAPES':
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/unet_cityscapes_1024x512_160k/model.pdparams',
    'DeepLabV3P_ResNet50_vd_CITYSCAPES':
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k/model.pdparams',
    'DeepLabV3P_ResNet101_vd_CITYSCAPES':
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet101_os8_cityscapes_769x769_80k/model.pdparams',
    'HRNet_HRNet_W18_CITYSCAPES':
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw18_cityscapes_1024x512_80k/model.pdparams',
    'HRNet_HRNet_W48_CITYSCAPES':
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw48_cityscapes_1024x512_80k/model.pdparams',
    'BiSeNetV2_CITYSCAPES':
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams',
    'FastSCNN_CITYSCAPES':
    'https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fastscnn_cityscapes_1024x1024_160k/model.pdparams'
}

imagenet_weights = {
    'ResNet18_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet18_pretrained.pdparams',
    'ResNet34_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet34_pretrained.pdparams',
    'ResNet50_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_pretrained.pdparams',
    'ResNet101_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet101_pretrained.pdparams',
    'ResNet152_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet101_pretrained.pdparams',
    'ResNet18_vd_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet18_vd_pretrained.pdparams',
    'ResNet34_vd_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet34_vd_pretrained.pdparams',
    'ResNet50_vd_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams',
    'ResNet50_vd_ssld_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams',
    'ResNet101_vd_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet101_vd_pretrained.pdparams',
    'ResNet101_vd_ssld_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet101_vd_ssld_pretrained.pdparams',
    'ResNet152_vd_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet152_vd_pretrained.pdparams',
    'ResNet200_vd_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet200_vd_pretrained.pdparams',
    'MobileNetV1_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_pretrained.pdparams',
    'MobileNetV1_x0_25_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_x0_25_pretrained.pdparams',
    'MobileNetV1_x0_5_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_x0_5_pretrained.pdparams',
    'MobileNetV1_x0_75_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_x0_75_pretrained.pdparams',
    'MobileNetV2_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_pretrained.pdparams',
    'MobileNetV2_x0_25_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_25_pretrained.pdparams',
    'MobileNetV2_x0_5_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_5_pretrained.pdparams',
    'MobileNetV2_x0_75_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_75_pretrained.pdparams',
    'MobileNetV2_x1_5_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x1_5_pretrained.pdparams',
    'MobileNetV2_x2_0_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x2_0_pretrained.pdparams',
    'MobileNetV3_small_x0_35_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x0_35_pretrained.pdparams',
    'MobileNetV3_small_x0_35_ssld_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x0_35_ssld_pretrained.pdparams',
    'MobileNetV3_small_x0_5_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x0_5_pretrained.pdparams',
    'MobileNetV3_small_x0_75_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x0_75_pretrained.pdparams',
    'MobileNetV3_small_x1_0_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x1_0_pretrained.pdparams',
    'MobileNetV3_small_x1_0_ssld_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x1_0_ssld_pretrained.pdparams',
    'MobileNetV3_small_x1_25_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x1_25_pretrained.pdparams',
    'MobileNetV3_large_x0_35_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_35_pretrained.pdparams',
    'MobileNetV3_large_x0_5_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams',
    'MobileNetV3_large_x0_75_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_75_pretrained.pdparams',
    'MobileNetV3_large_x1_0_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_pretrained.pdparams',
    'MobileNetV3_large_x1_25_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_25_pretrained.pdparams',
    'MobileNetV3_large_x1_0_ssld_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_ssld_pretrained.pdparams',
    'AlexNet_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/AlexNet_pretrained.pdparams',
    'DarkNet53_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DarkNet53_pretrained.pdparams',
    'DenseNet121_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams',
    'DenseNet161_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet161_pretrained.pdparams',
    'DenseNet169_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet169_pretrained.pdparams',
    'DenseNet201_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet201_pretrained.pdparams',
    'DenseNet264_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet264_pretrained.pdparams',
    'HRNet_W18_C_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W18_C_pretrained.pdparams',
    'HRNet_W30_C_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W30_C_pretrained.pdparams',
    'HRNet_W32_C_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W32_C_pretrained.pdparams',
    'HRNet_W40_C_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W40_C_pretrained.pdparams',
    'HRNet_W44_C_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W44_C_pretrained.pdparams',
    'HRNet_W48_C_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W48_C_pretrained.pdparams',
    'HRNet_W64_C_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W64_C_pretrained.pdparams',
    'Xception41_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception41_pretrained.pdparams',
    'Xception65_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception65_pretrained.pdparams',
    'Xception71_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception71_pretrained.pdparams',
    'ShuffleNetV2_x0_25_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_25_pretrained.pdparams',
    'ShuffleNetV2_x0_33_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_33_pretrained.pdparams',
    'ShuffleNetV2_x0_5_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_5_pretrained.pdparams',
    'ShuffleNetV2_x1_0_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_0_pretrained.pdparams',
    'ShuffleNetV2_x1_5_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_5_pretrained.pdparams',
    'ShuffleNetV2_x2_0_IMAGENET':
    'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x2_0_pretrained.pdparams',
    'FasterRCNN_ResNet50_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_cos_pretrained.pdparams',
    'FasterRCNN_ResNet50_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_cos_pretrained.pdparams',
    'FasterRCNN_ResNet50_vd_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_pretrained.pdparams',
    'FasterRCNN_ResNet50_vd_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_pretrained.pdparams',
    'FasterRCNN_ResNet50_vd_ssld_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_v2_pretrained.pdparams',
    'FasterRCNN_ResNet34_vd_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet34_vd_pretrained.pdparams',
    'FasterRCNN_ResNet34_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet34_pretrained.pdparams',
    'FasterRCNN_ResNet101_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet101_pretrained.pdparams',
    'FasterRCNN_ResNet101_vd_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet101_vd_pretrained.pdparams',
    'FasterRCNN_HRNet_W18_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/HRNet_W18_C_pretrained.pdparams',
    'YOLOv3_ResNet50_vd_dcn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_pretrained.pdparams',
    'YOLOv3_ResNet34_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet34_pretrained.pdparams',
    'YOLOv3_MobileNetV1_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/MobileNetV1_pretrained.pdparams',
    'YOLOv3_MobileNetV1_ssld_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/MobileNetV1_ssld_pretrained.pdparams',
    'YOLOv3_MobileNetV3_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/MobileNetV3_large_x1_0_ssld_pretrained.pdparams',
    'YOLOv3_MobileNetV3_ssld_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/MobileNetV3_large_x1_0_ssld_pretrained.pdparams',
    'YOLOv3_DarkNet53_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/DarkNet53_pretrained.pdparams',
    'PPYOLO_ResNet50_vd_dcn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_pretrained.pdparams',
    'PPYOLO_ResNet18_vd_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet18_vd_pretrained.pdparams',
    'PPYOLO_MobileNetV3_large_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/MobileNetV3_large_x1_0_ssld_pretrained.pdparams',
    'PPYOLO_MobileNetV3_small_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/MobileNetV3_small_x1_0_ssld_pretrained.pdparams',
    'PPYOLOv2_ResNet50_vd_dcn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_pretrained.pdparams',
    'PPYOLOv2_ResNet101_vd_dcn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet101_vd_ssld_pretrained.pdparams',
    'PPYOLOTiny_MobileNetV3_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams',
    'MaskRCNN_ResNet50_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_cos_pretrained.pdparams',
    'MaskRCNN_ResNet50_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_cos_pretrained.pdparams',
    'MaskRCNN_ResNet50_vd_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_pretrained.pdparams',
    'MaskRCNN_ResNet50_vd_ssld_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_v2_pretrained.pdparams',
    'MaskRCNN_ResNet101_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet101_pretrained.pdparams',
    'MaskRCNN_ResNet101_vd_fpn_IMAGENET':
    'https://paddledet.bj.bcebos.com/models/pretrained/ResNet101_vd_pretrained.pdparams',
    'DeepLabV3P_ResNet50_vd_IMAGENET':
    'https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz',
    'DeepLabV3P_ResNet101_vd_IMAGENET':
    'https://bj.bcebos.com/paddleseg/dygraph/resnet101_vd_ssld.tar.gz'
}

pascalvoc_weights = {
    'DeepLabV3P_ResNet50_vd_PascalVOC':
    'https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/deeplabv3p_resnet50_os8_voc12aug_512x512_40k/model.pdparams',
    'DeepLabV3P_ResNet101_vd_PascalVOC':
    'https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/deeplabv3p_resnet101_os8_voc12aug_512x512_40k/model.pdparams',
    'HRNet_HRNet_W18_PascalVOC':
    'https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/fcn_hrnetw18_voc12aug_512x512_40k/model.pdparams',
    'HRNet_HRNet_W48_PascalVOC':
    'https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/fcn_hrnetw48_voc12aug_512x512_40k/model.pdparams',
    'YOLOv3_MobileNetV1_PascalVOC':
    'https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_voc.pdparams',
    'YOLOv3_MobileNetV1_ssld_PascalVOC':
    'https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_ssld_270e_voc.pdparams',
    'YOLOv3_MobileNetV3_PascalVOC':
    'https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_voc.pdparams',
    'YOLOv3_MobileNetV3_ssld_PascalVOC':
    'https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_ssld_270e_voc.pdparams'
}

coco_weights = {
    'YOLOv3_MobileNetV1_COCO':
    'https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams',
    'YOLOv3_MobileNetV1_ssld_COCO':
    'https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_ssld_270e_coco.pdparams',
    'YOLOv3_DarkNet53_COCO':
    'https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams',
    'YOLOv3_ResNet50_vd_dcn_COCO':
    'https://paddledet.bj.bcebos.com/models/yolov3_r50vd_dcn_270e_coco.pdparams',
    'YOLOv3_ResNet34_COCO':
    'https://paddledet.bj.bcebos.com/models/yolov3_r34_270e_coco.pdparams',
    'YOLOv3_MobileNetV3_COCO':
    'https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams',
    'FasterRCNN_ResNet50_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_2x_coco.pdparams',
    'FasterRCNN_ResNet50_COCO':
    'https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_1x_coco.pdparams',
    'FasterRCNN_ResNet50_vd_COCO':
    'https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_1x_coco.pdparams',
    'FasterRCNN_ResNet50_vd_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_2x_coco.pdparams',
    'FasterRCNN_ResNet50_vd_ssld_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_ssld_fpn_2x_coco.pdparams',
    'FasterRCNN_ResNet34_vd_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/faster_rcnn_r34_vd_fpn_1x_coco.pdparams',
    'FasterRCNN_ResNet34_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/faster_rcnn_r34_fpn_1x_coco.pdparams',
    'FasterRCNN_ResNet101_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_fpn_2x_coco.pdparams',
    'FasterRCNN_ResNet101_vd_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_vd_fpn_1x_coco.pdparams',
    'FasterRCNN_HRNet_W18_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/faster_rcnn_hrnetv2p_w18_2x_coco.pdparams',
    'PPYOLO_ResNet50_vd_dcn_COCO':
    'https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams',
    'PPYOLO_ResNet18_vd_COCO':
    'https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams',
    'PPYOLO_MobileNetV3_large_COCO':
    'https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams',
    'PPYOLO_MobileNetV3_small_COCO':
    'https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_small_coco.pdparams',
    'PPYOLOv2_ResNet50_vd_dcn_COCO':
    'https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams',
    'PPYOLOv2_ResNet101_vd_dcn_COCO':
    'https://paddledet.bj.bcebos.com/models/ppyolov2_r101vd_dcn_365e_coco.pdparams',
    'PPYOLOTiny_MobileNetV3_COCO':
    'https://paddledet.bj.bcebos.com/models/ppyolo_tiny_650e_coco.pdparams',
    'MaskRCNN_ResNet50_COCO':
    'https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_2x_coco.pdparams',
    'MaskRCNN_ResNet50_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_fpn_2x_coco.pdparams',
    'MaskRCNN_ResNet50_vd_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_2x_coco.pdparams',
    'MaskRCNN_ResNet50_vd_ssld_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_ssld_2x_coco.pdparams',
    'MaskRCNN_ResNet101_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/mask_rcnn_r101_fpn_1x_coco.pdparams',
    'MaskRCNN_ResNet101_vd_fpn_COCO':
    'https://paddledet.bj.bcebos.com/models/mask_rcnn_r101_vd_fpn_1x_coco.pdparams'
}


def get_pretrain_weights(flag, class_name, save_dir, backbone_name=None):
    if flag is None:
        return None
    elif osp.isdir(flag):
        return flag
    elif osp.isfile(flag):
        return flag

    # TODO: check flag
    new_save_dir = save_dir
    if backbone_name is not None:
        weights_key = "{}_{}_{}".format(class_name, backbone_name, flag)
    else:
        weights_key = "{}_{}".format(class_name, flag)
    if flag == 'CITYSCAPES':
        url = cityscapes_weights[weights_key]
    elif flag == 'IMAGENET':
        url = imagenet_weights[weights_key]
    elif flag == 'PascalVOC':
        url = pascalvoc_weights[weights_key]
    elif flag == 'COCO':
        url = coco_weights[weights_key]
    else:
        raise ValueError('Given pretrained weights {} is undefined.'.format(
            flag))
    fname = download_and_decompress(url, path=new_save_dir)
    if osp.isdir(fname):
        fname = glob.glob(osp.join(fname, '*.pdparams'))[0]
    return fname


def load_pretrain_weights(model, pretrain_weights=None, model_name=None):
    if pretrain_weights is not None:
        logging.info(
            'Loading pretrained model from {}'.format(pretrain_weights),
            use_color=True)

        if os.path.exists(pretrain_weights):
            param_state_dict = paddle.load(pretrain_weights)
            model_state_dict = model.state_dict()
            # hack: fit for faster rcnn. Pretrain weights contain prefix of 'backbone'
            # while res5 module is located in bbox_head.head. Replace the prefix of
            # res5 with 'bbox_head.head' to load pretrain weights correctly.
            for k in list(param_state_dict.keys()):
                if 'backbone.res5' in k:
                    new_k = k.replace('backbone', 'bbox_head.head')
                    if new_k in model_state_dict:
                        value = param_state_dict.pop(k)
                        param_state_dict[new_k] = value
            num_params_loaded = 0
            for k in model_state_dict:
                if k not in param_state_dict:
                    logging.warning("{} is not in pretrained model".format(k))
                elif list(param_state_dict[k].shape) != list(model_state_dict[
                        k].shape):
                    logging.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, param_state_dict[k].shape, model_state_dict[
                            k].shape))
                else:
                    model_state_dict[k] = param_state_dict[k]
                    num_params_loaded += 1
            model.set_state_dict(model_state_dict)
            logging.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded, len(model_state_dict), model_name))
        else:
            raise ValueError('The pretrained model directory is not Found: {}'.
                             format(pretrain_weights))
    else:
        logging.info(
            'No pretrained model to load, {} will be trained from scratch.'.
            format(model_name))


def load_optimizer(optimizer, state_dict_path):
    logging.info("Loading optimizer from {}".format(state_dict_path))
    optim_state_dict = paddle.load(state_dict_path)
    if 'last_epoch' in optim_state_dict:
        optim_state_dict.pop('last_epoch')
    optimizer.set_state_dict(optim_state_dict)


def load_checkpoint(model, optimizer, model_name, checkpoint):
    logging.info("Loading checkpoint from {}".format(checkpoint))
    load_pretrain_weights(
        model,
        pretrain_weights=osp.join(checkpoint, 'model.pdparams'),
        model_name=model_name)
    load_optimizer(
        optimizer, state_dict_path=osp.join(checkpoint, "model.pdopt"))
