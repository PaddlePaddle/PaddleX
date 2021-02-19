# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import os.path as osp
import paddle.fluid as fluid
import paddlex

sensitivities_data = {
    'AlexNet':
    'https://bj.bcebos.com/paddlex/slim_prune/alexnet_sensitivities.data',
    'ResNet18':
    'https://bj.bcebos.com/paddlex/slim_prune/resnet18.sensitivities',
    'ResNet34':
    'https://bj.bcebos.com/paddlex/slim_prune/resnet34.sensitivities',
    'ResNet50':
    'https://bj.bcebos.com/paddlex/slim_prune/resnet50.sensitivities',
    'ResNet101':
    'https://bj.bcebos.com/paddlex/slim_prune/resnet101.sensitivities',
    'ResNet50_vd':
    'https://bj.bcebos.com/paddlex/slim_prune/resnet50vd.sensitivities',
    'ResNet101_vd':
    'https://bj.bcebos.com/paddlex/slim_prune/resnet101vd.sensitivities',
    'DarkNet53':
    'https://bj.bcebos.com/paddlex/slim_prune/darknet53.sensitivities',
    'MobileNetV1':
    'https://bj.bcebos.com/paddlex/slim_prune/mobilenetv1.sensitivities',
    'MobileNetV2':
    'https://bj.bcebos.com/paddlex/slim_prune/mobilenetv2.sensitivities',
    'MobileNetV3_large':
    'https://bj.bcebos.com/paddlex/slim_prune/mobilenetv3_large.sensitivities',
    'MobileNetV3_small':
    'https://bj.bcebos.com/paddlex/slim_prune/mobilenetv3_small.sensitivities',
    'MobileNetV3_large_ssld':
    'https://bj.bcebos.com/paddlex/slim_prune/mobilenetv3_large_ssld_sensitivities.data',
    'MobileNetV3_small_ssld':
    'https://bj.bcebos.com/paddlex/slim_prune/mobilenetv3_small_ssld_sensitivities.data',
    'DenseNet121':
    'https://bj.bcebos.com/paddlex/slim_prune/densenet121.sensitivities',
    'DenseNet161':
    'https://bj.bcebos.com/paddlex/slim_prune/densenet161.sensitivities',
    'DenseNet201':
    'https://bj.bcebos.com/paddlex/slim_prune/densenet201.sensitivities',
    'Xception41':
    'https://bj.bcebos.com/paddlex/slim_prune/xception41.sensitivities',
    'Xception65':
    'https://bj.bcebos.com/paddlex/slim_prune/xception65.sensitivities',
    'ShuffleNetV2':
    'https://bj.bcebos.com/paddlex/slim_prune/shufflenetv2_sensitivities.data',
    'YOLOv3_MobileNetV1':
    'https://bj.bcebos.com/paddlex/slim_prune/yolov3_mobilenetv1.sensitivities',
    'YOLOv3_MobileNetV3_large':
    'https://bj.bcebos.com/paddlex/slim_prune/yolov3_mobilenetv3.sensitivities',
    'YOLOv3_DarkNet53':
    'https://bj.bcebos.com/paddlex/slim_prune/yolov3_darknet53.sensitivities',
    'YOLOv3_ResNet34':
    'https://bj.bcebos.com/paddlex/slim_prune/yolov3_resnet34.sensitivities',
    'UNet': 'https://bj.bcebos.com/paddlex/slim_prune/unet.sensitivities',
    'DeepLabv3p_MobileNetV2_x0.25':
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_mobilenetv2_x0.25_no_aspp_decoder.sensitivities',
    'DeepLabv3p_MobileNetV2_x0.5':
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_mobilenetv2_x0.5_no_aspp_decoder.sensitivities',
    'DeepLabv3p_MobileNetV2_x1.0':
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_mobilenetv2_x1.0_no_aspp_decoder.sensitivities',
    'DeepLabv3p_MobileNetV2_x1.5':
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_mobilenetv2_x1.5_no_aspp_decoder.sensitivities',
    'DeepLabv3p_MobileNetV2_x2.0':
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_mobilenetv2_x2.0_no_aspp_decoder.sensitivities',
    'DeepLabv3p_MobileNetV2_x0.25_aspp_decoder':
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_mobilenetv2_x0.25_with_aspp_decoder.sensitivities',
    'DeepLabv3p_MobileNetV2_x0.5_aspp_decoder':
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_mobilenetv2_x0.5_with_aspp_decoder.sensitivities',
    'DeepLabv3p_MobileNetV2_x1.0_aspp_decoder':
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_mobilenetv2_x1.0_with_aspp_decoder.sensitivities',
    'DeepLabv3p_MobileNetV2_x1.5_aspp_decoder':
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_mobilenetv2_x1.5_with_aspp_decoder.sensitivities',
    'DeepLabv3p_MobileNetV2_x2.0_aspp_decoder':
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_mobilenetv2_x2.0_with_aspp_decoder.sensitivities',
    'DeepLabv3p_Xception65_aspp_decoder':
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_xception65_with_aspp_decoder.sensitivities',
    'DeepLabv3p_Xception41_aspp_decoder':
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_xception41_with_aspp_decoder.sensitivities',
    'HRNet_W18_Seg':
    'https://bj.bcebos.com/paddlex/slim_prune/hrnet_w18.sensitivities',
    'HRNet_W30_Seg':
    'https://bj.bcebos.com/paddlex/slim_prune/hrnet_w30.sensitivities',
    'HRNet_W32_Seg':
    'https://bj.bcebos.com/paddlex/slim_prune/hrnet_w32.sensitivities',
    'HRNet_W40_Seg':
    'https://bj.bcebos.com/paddlex/slim_prune/hrnet_w40.sensitivities',
    'HRNet_W44_Seg':
    'https://bj.bcebos.com/paddlex/slim_prune/hrnet_w44.sensitivities',
    'HRNet_W48_Seg':
    'https://bj.bcebos.com/paddlex/slim_prune/hrnet_w48.sensitivities',
    'HRNet_W64_Seg':
    'https://bj.bcebos.com/paddlex/slim_prune/hrnet_w64.sensitivities',
    'FastSCNN':
    'https://bj.bcebos.com/paddlex/slim_prune/fast_scnn.sensitivities'
}


def get_sensitivities(flag, model, save_dir):
    model_name = model.__class__.__name__
    model_type = model_name
    if hasattr(model, 'backbone'):
        model_type = model_name + '_' + model.backbone
    if model_type.startswith('DeepLabv3p_Xception'):
        model_type = model_type + '_' + 'aspp' + '_' + 'decoder'
    elif hasattr(model, 'encoder_with_aspp') or hasattr(model,
                                                        'enable_decoder'):
        model_type = model_type + '_' + 'aspp' + '_' + 'decoder'
    if model_type.startswith('HRNet') and model.model_type == 'segmenter':
        model_type = '{}_W{}_Seg'.format(model_type, model.width)
    if osp.isfile(flag):
        return flag
    elif flag == 'DEFAULT':
        assert model_type in sensitivities_data, "There is not sensitivities data file for {}, you may need to calculate it by your self.".format(
            model_type)
        url = sensitivities_data[model_type]
        fname = osp.split(url)[-1]
        if getattr(paddlex, 'gui_mode', False):
            paddlex.utils.download(url, path=save_dir)
            return osp.join(save_dir, fname)

        import paddlehub as hub
        try:
            hub.download(fname, save_path=save_dir)
        except Exception as e:
            if isinstance(e, hub.ResourceNotFoundError):
                raise Exception("Resource for model {}(key='{}') not found".
                                format(model_type, fname))
            elif isinstance(e, hub.ServerConnectionError):
                raise Exception(
                    "Cannot get reource for model {}(key='{}'), please check your internet connection"
                    .format(model_type, fname))
            else:
                raise Exception(
                    "Unexpected error, please make sure paddlehub >= 1.6.2 {}".
                    format(str(e)))
        return osp.join(save_dir, fname)
    else:
        raise Exception(
            "sensitivities need to be defined as directory path or `DEFAULT`(download sensitivities automatically)."
        )


def get_prune_params(model):
    prune_names = []
    model_type = model.__class__.__name__
    if model_type == 'BaseClassifier':
        model_type = model.model_name
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        model_type += ('_' + backbone)
    program = model.test_prog
    if model_type.startswith('ResNet') or \
            model_type.startswith('DenseNet') or \
            model_type.startswith('DarkNet') or \
            model_type.startswith('AlexNet') or \
            model_type.startswith('ShuffleNetV2'):
        for block in program.blocks:
            for param in block.all_parameters():
                pd_var = model.scope.find_var(param.name)
                try:
                    pd_param = pd_var.get_tensor()
                    if len(np.array(pd_param).shape) == 4:
                        prune_names.append(param.name)
                except Exception as e:
                    print("None Tensor Name: ", param.name)
                    print("Error message: {}".format(e))
        if model_type == 'AlexNet':
            prune_names.remove('conv5_weights')
        if model_type == 'ShuffleNetV2':
            not_prune_names = [
                'stage_2_1_conv5_weights',
                'stage_2_1_conv3_weights',
                'stage_2_2_conv3_weights',
                'stage_2_3_conv3_weights',
                'stage_2_4_conv3_weights',
                'stage_3_1_conv5_weights',
                'stage_3_1_conv3_weights',
                'stage_3_2_conv3_weights',
                'stage_3_3_conv3_weights',
                'stage_3_4_conv3_weights',
                'stage_3_5_conv3_weights',
                'stage_3_6_conv3_weights',
                'stage_3_7_conv3_weights',
                'stage_3_8_conv3_weights',
                'stage_4_1_conv5_weights',
                'stage_4_1_conv3_weights',
                'stage_4_2_conv3_weights',
                'stage_4_3_conv3_weights',
                'stage_4_4_conv3_weights',
            ]
            for name in not_prune_names:
                prune_names.remove(name)
    elif model_type == "MobileNetV1":
        prune_names.append("conv1_weights")
        for param in program.global_block().all_parameters():
            if "_sep_weights" in param.name:
                prune_names.append(param.name)
    elif model_type == "MobileNetV2":
        for param in program.global_block().all_parameters():
            if 'weight' not in param.name \
                    or 'dwise' in param.name \
                    or 'fc' in param.name :
                continue
            prune_names.append(param.name)
    elif model_type.startswith("MobileNetV3"):
        if model_type.startswith('MobileNetV3_small'):
            expand_prune_id = [3, 4]
        else:
            expand_prune_id = [2, 3, 4, 8, 9, 11]
        for param in program.global_block().all_parameters():
            if ('expand_weights' in param.name and \
                    int(param.name.split('_')[0][4:]) in expand_prune_id)\
                    or 'linear_weights' in param.name \
                    or 'se_1_weights' in param.name:
                prune_names.append(param.name)
    elif model_type.startswith('Xception') or \
            model_type.startswith('DeepLabv3p_Xception'):
        params_not_prune = [
            'weights',
            'xception_{}/exit_flow/block2/separable_conv3/pointwise/weights'.
            format(model_type[-2:]), 'encoder/concat/weights',
            'decoder/concat/weights'
        ]
        for param in program.global_block().all_parameters():
            if 'weight' not in param.name \
                    or 'dwise' in param.name \
                    or 'depthwise' in param.name \
                    or 'logit' in param.name:
                continue
            if param.name in params_not_prune:
                continue
            prune_names.append(param.name)
    elif model_type.startswith('YOLOv3'):
        for block in program.blocks:
            for param in block.all_parameters():
                if 'weights' in param.name and 'yolo_block' in param.name:
                    prune_names.append(param.name)
    elif model_type.startswith('UNet'):
        for param in program.global_block().all_parameters():
            if 'weight' not in param.name:
                continue
            if 'logit' in param.name:
                continue
            prune_names.append(param.name)
        params_not_prune = [
            'encode/block4/down/conv1/weights',
            'encode/block3/down/conv1/weights',
            'encode/block2/down/conv1/weights', 'encode/block1/conv1/weights'
        ]
        for i in params_not_prune:
            if i in prune_names:
                prune_names.remove(i)

    elif model_type.startswith('HRNet') and model.model_type == 'segmenter':
        for param in program.global_block().all_parameters():
            if 'weight' not in param.name:
                continue
            prune_names.append(param.name)
        params_not_prune = ['conv-1_weights']
        for i in params_not_prune:
            if i in prune_names:
                prune_names.remove(i)

    elif model_type.startswith('FastSCNN'):
        for param in program.global_block().all_parameters():
            if 'weight' not in param.name:
                continue
            if 'dwise' in param.name or 'depthwise' in param.name or 'logit' in param.name:
                continue
            prune_names.append(param.name)
        params_not_prune = ['classifier/weights']
        for i in params_not_prune:
            if i in prune_names:
                prune_names.remove(i)

    elif model_type.startswith('DeepLabv3p'):
        if model_type.lower() == "deeplabv3p_mobilenetv3_large_x1_0_ssld":
            params_not_prune = [
                'last_1x1_conv_weights', 'conv14_se_2_weights',
                'conv16_depthwise_weights', 'conv13_depthwise_weights',
                'conv15_se_2_weights', 'conv2_depthwise_weights',
                'conv6_depthwise_weights', 'conv8_depthwise_weights',
                'fc_weights', 'conv3_depthwise_weights', 'conv7_se_2_weights',
                'conv16_expand_weights', 'conv16_se_2_weights',
                'conv10_depthwise_weights', 'conv11_depthwise_weights',
                'conv15_expand_weights', 'conv5_expand_weights',
                'conv15_depthwise_weights', 'conv14_depthwise_weights',
                'conv12_se_2_weights', 'conv1_weights',
                'conv13_expand_weights', 'conv_last_weights',
                'conv12_depthwise_weights', 'conv13_se_2_weights',
                'conv12_expand_weights', 'conv5_depthwise_weights',
                'conv6_se_2_weights', 'conv10_expand_weights',
                'conv9_depthwise_weights', 'conv6_expand_weights',
                'conv5_se_2_weights', 'conv14_expand_weights',
                'conv4_depthwise_weights', 'conv7_expand_weights',
                'conv7_depthwise_weights', 'encoder/aspp0/weights',
                'decoder/merge/weights', 'encoder/image_pool/weights',
                'decoder/weights'
            ]
        for param in program.global_block().all_parameters():
            if 'weight' not in param.name:
                continue
            if 'dwise' in param.name or 'depthwise' in param.name or 'logit' in param.name:
                continue
            if model_type.lower() == "deeplabv3p_mobilenetv3_large_x1_0_ssld":
                if param.name in params_not_prune:
                    continue
            prune_names.append(param.name)
        params_not_prune = [
            'xception_{}/exit_flow/block2/separable_conv3/pointwise/weights'.
            format(model_type[-2:]), 'encoder/concat/weights',
            'decoder/concat/weights'
        ]
        if model.encoder_with_aspp == True:
            params_not_prune.append(
                'xception_{}/exit_flow/block2/separable_conv3/pointwise/weights'
                .format(model_type[-2:]))
            params_not_prune.append('conv8_1_linear_weights')
        for i in params_not_prune:
            if i in prune_names:
                prune_names.remove(i)
    elif 'RCNN' in model_type:
        for block in program.blocks:
            for param in block.all_parameters():
                pd_var = model.scope.find_var(param.name)
                pd_param = pd_var.get_tensor()
                if len(np.array(pd_param).shape) == 4:
                    if 'fpn' in param.name or 'rpn' in param.name or 'fc' in param.name or 'cls' in param.name or 'bbox' in param.name:
                        continue
                    prune_names.append(param.name)
    else:
        raise Exception('The {} is not implement yet!'.format(model_type))
    return prune_names
