# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import sys
import inspect

from paddlex.ppcls.arch.backbone.legendary_models.mobilenet_v1 import MobileNetV1_x0_25, MobileNetV1_x0_5, MobileNetV1_x0_75, MobileNetV1
from paddlex.ppcls.arch.backbone.legendary_models.mobilenet_v3 import MobileNetV3_small_x0_35, MobileNetV3_small_x0_5, MobileNetV3_small_x0_75, MobileNetV3_small_x1_0, MobileNetV3_small_x1_25, MobileNetV3_large_x0_35, MobileNetV3_large_x0_5, MobileNetV3_large_x0_75, MobileNetV3_large_x1_0, MobileNetV3_large_x1_25
from paddlex.ppcls.arch.backbone.legendary_models.resnet import ResNet18, ResNet18_vd, ResNet34, ResNet34_vd, ResNet50, ResNet50_vd, ResNet101, ResNet101_vd, ResNet152, ResNet152_vd, ResNet200_vd
from paddlex.ppcls.arch.backbone.legendary_models.vgg import VGG11, VGG13, VGG16, VGG19
from paddlex.ppcls.arch.backbone.legendary_models.inception_v3 import InceptionV3
from paddlex.ppcls.arch.backbone.legendary_models.hrnet import HRNet_W18_C, HRNet_W30_C, HRNet_W32_C, HRNet_W40_C, HRNet_W44_C, HRNet_W48_C, HRNet_W60_C, HRNet_W64_C, SE_HRNet_W64_C
from paddlex.ppcls.arch.backbone.legendary_models.pp_lcnet import PPLCNet_x0_25, PPLCNet_x0_35, PPLCNet_x0_5, PPLCNet_x0_75, PPLCNet_x1_0, PPLCNet_x1_5, PPLCNet_x2_0, PPLCNet_x2_5
from paddlex.ppcls.arch.backbone.legendary_models.esnet import ESNet_x0_25, ESNet_x0_5, ESNet_x0_75, ESNet_x1_0

from paddlex.ppcls.arch.backbone.model_zoo.resnet_vc import ResNet50_vc
from paddlex.ppcls.arch.backbone.model_zoo.resnext import ResNeXt50_32x4d, ResNeXt50_64x4d, ResNeXt101_32x4d, ResNeXt101_64x4d, ResNeXt152_32x4d, ResNeXt152_64x4d
from paddlex.ppcls.arch.backbone.model_zoo.resnext_vd import ResNeXt50_vd_32x4d, ResNeXt50_vd_64x4d, ResNeXt101_vd_32x4d, ResNeXt101_vd_64x4d, ResNeXt152_vd_32x4d, ResNeXt152_vd_64x4d
from paddlex.ppcls.arch.backbone.model_zoo.res2net import Res2Net50_26w_4s, Res2Net50_14w_8s
from paddlex.ppcls.arch.backbone.model_zoo.res2net_vd import Res2Net50_vd_26w_4s, Res2Net101_vd_26w_4s, Res2Net200_vd_26w_4s
from paddlex.ppcls.arch.backbone.model_zoo.se_resnet_vd import SE_ResNet18_vd, SE_ResNet34_vd, SE_ResNet50_vd
from paddlex.ppcls.arch.backbone.model_zoo.se_resnext_vd import SE_ResNeXt50_vd_32x4d, SE_ResNeXt50_vd_32x4d, SENet154_vd
from paddlex.ppcls.arch.backbone.model_zoo.se_resnext import SE_ResNeXt50_32x4d, SE_ResNeXt101_32x4d, SE_ResNeXt152_64x4d
from paddlex.ppcls.arch.backbone.model_zoo.dpn import DPN68, DPN92, DPN98, DPN107, DPN131
from paddlex.ppcls.arch.backbone.model_zoo.densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201, DenseNet264
from paddlex.ppcls.arch.backbone.model_zoo.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7, EfficientNetB0_small
from paddlex.ppcls.arch.backbone.model_zoo.resnest import ResNeSt50_fast_1s1x64d, ResNeSt50, ResNeSt101
from paddlex.ppcls.arch.backbone.model_zoo.googlenet import GoogLeNet
from paddlex.ppcls.arch.backbone.model_zoo.mobilenet_v2 import MobileNetV2_x0_25, MobileNetV2_x0_5, MobileNetV2_x0_75, MobileNetV2, MobileNetV2_x1_5, MobileNetV2_x2_0
from paddlex.ppcls.arch.backbone.model_zoo.shufflenet_v2 import ShuffleNetV2_x0_25, ShuffleNetV2_x0_33, ShuffleNetV2_x0_5, ShuffleNetV2_x1_0, ShuffleNetV2_x1_5, ShuffleNetV2_x2_0, ShuffleNetV2_swish
from paddlex.ppcls.arch.backbone.model_zoo.ghostnet import GhostNet_x0_5, GhostNet_x1_0, GhostNet_x1_3
from paddlex.ppcls.arch.backbone.model_zoo.alexnet import AlexNet
from paddlex.ppcls.arch.backbone.model_zoo.inception_v4 import InceptionV4
from paddlex.ppcls.arch.backbone.model_zoo.xception import Xception41, Xception65, Xception71
from paddlex.ppcls.arch.backbone.model_zoo.xception_deeplab import Xception41_deeplab, Xception65_deeplab
from paddlex.ppcls.arch.backbone.model_zoo.resnext101_wsl import ResNeXt101_32x8d_wsl, ResNeXt101_32x16d_wsl, ResNeXt101_32x32d_wsl, ResNeXt101_32x48d_wsl
from paddlex.ppcls.arch.backbone.model_zoo.squeezenet import SqueezeNet1_0, SqueezeNet1_1
from paddlex.ppcls.arch.backbone.model_zoo.darknet import DarkNet53
from paddlex.ppcls.arch.backbone.model_zoo.regnet import RegNetX_200MF, RegNetX_4GF, RegNetX_32GF, RegNetY_200MF, RegNetY_4GF, RegNetY_32GF
from paddlex.ppcls.arch.backbone.model_zoo.vision_transformer import ViT_small_patch16_224, ViT_base_patch16_224, ViT_base_patch16_384, ViT_base_patch32_384, ViT_large_patch16_224, ViT_large_patch16_384, ViT_large_patch32_384, ViT_huge_patch16_224, ViT_huge_patch32_384
from paddlex.ppcls.arch.backbone.model_zoo.distilled_vision_transformer import DeiT_tiny_patch16_224, DeiT_small_patch16_224, DeiT_base_patch16_224, DeiT_tiny_distilled_patch16_224, DeiT_small_distilled_patch16_224, DeiT_base_distilled_patch16_224, DeiT_base_patch16_384, DeiT_base_distilled_patch16_384
from paddlex.ppcls.arch.backbone.model_zoo.swin_transformer import SwinTransformer_tiny_patch4_window7_224, SwinTransformer_small_patch4_window7_224, SwinTransformer_base_patch4_window7_224, SwinTransformer_base_patch4_window12_384, SwinTransformer_large_patch4_window7_224, SwinTransformer_large_patch4_window12_384
from paddlex.ppcls.arch.backbone.model_zoo.mixnet import MixNet_S, MixNet_M, MixNet_L
from paddlex.ppcls.arch.backbone.model_zoo.rexnet import ReXNet_1_0, ReXNet_1_3, ReXNet_1_5, ReXNet_2_0, ReXNet_3_0
from paddlex.ppcls.arch.backbone.model_zoo.gvt import pcpvt_small, pcpvt_base, pcpvt_large, alt_gvt_small, alt_gvt_base, alt_gvt_large
from paddlex.ppcls.arch.backbone.model_zoo.levit import LeViT_128S, LeViT_128, LeViT_192, LeViT_256, LeViT_384
from paddlex.ppcls.arch.backbone.model_zoo.dla import DLA34, DLA46_c, DLA46x_c, DLA60, DLA60x, DLA60x_c, DLA102, DLA102x, DLA102x2, DLA169
from paddlex.ppcls.arch.backbone.model_zoo.rednet import RedNet26, RedNet38, RedNet50, RedNet101, RedNet152
from paddlex.ppcls.arch.backbone.model_zoo.tnt import TNT_small
from paddlex.ppcls.arch.backbone.model_zoo.hardnet import HarDNet68, HarDNet85, HarDNet39_ds, HarDNet68_ds
from paddlex.ppcls.arch.backbone.model_zoo.cspnet import CSPDarkNet53
from paddlex.ppcls.arch.backbone.variant_models.resnet_variant import ResNet50_last_stage_stride1
from paddlex.ppcls.arch.backbone.variant_models.vgg_variant import VGG19Sigmoid
from paddlex.ppcls.arch.backbone.variant_models.pp_lcnet_variant import PPLCNet_x2_5_Tanh


def get_apis():
    current_func = sys._getframe().f_code.co_name
    current_module = sys.modules[__name__]
    api = []
    for _, obj in inspect.getmembers(current_module,
                                     inspect.isclass) + inspect.getmembers(
                                         current_module, inspect.isfunction):
        api.append(obj.__name__)
    api.remove(current_func)
    return api


__all__ = get_apis()
