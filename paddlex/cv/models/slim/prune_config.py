# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
#import paddlehub as hub
import paddlex

sensitivities_data = {
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
    'YOLOv3_MobileNetV1':
    'https://bj.bcebos.com/paddlex/slim_prune/yolov3_mobilenetv1.sensitivities',
    'YOLOv3_MobileNetV3_large':
    'https://bj.bcebos.com/paddlex/slim_prune/yolov3_mobilenetv3.sensitivities',
    'YOLOv3_DarkNet53':
    'https://bj.bcebos.com/paddlex/slim_prune/yolov3_darknet53.sensitivities',
    'YOLOv3_ResNet34':
    'https://bj.bcebos.com/paddlex/slim_prune/yolov3_resnet34.sensitivities',
    'UNet':
    'https://bj.bcebos.com/paddlex/slim_prune/unet.sensitivities',
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
    'https://bj.bcebos.com/paddlex/slim_prune/deeplab_xception41_with_aspp_decoder.sensitivities'
}


def get_sensitivities(flag, model, save_dir):
    model_name = model.__class__.__name__
    model_type = model_name
    if hasattr(model, 'backbone'):
        model_type = model_name + '_' + model.backbone
    if model_type.startswith('DeepLabv3p_Xception'):
        model_type = model_type + '_' + 'aspp' + '_' + 'decoder'
    elif hasattr(model, 'encoder_with_aspp') or hasattr(
            model, 'enable_decoder'):
        model_type = model_type + '_' + 'aspp' + '_' + 'decoder'
    if osp.isfile(flag):
        return flag
    elif flag == 'DEFAULT':
        assert model_type in sensitivities_data, "There is not sensitivities data file for {}, you may need to calculate it by your self.".format(
            model_type)
        url = sensitivities_data[model_type]
        fname = osp.split(url)[-1]
        paddlex.utils.download(url, path=save_dir)
        return osp.join(save_dir, fname)


#        try:
#            hub.download(fname, save_path=save_dir)
#        except Exception as e:
#            if isinstance(e, hub.ResourceNotFoundError):
#                raise Exception(
#                    "Resource for model {}(key='{}') not found".format(
#                        model_type, fname))
#            elif isinstance(e, hub.ServerConnectionError):
#                raise Exception(
#                    "Cannot get reource for model {}(key='{}'), please check your internet connecgtion"
#                    .format(model_type, fname))
#            else:
#                raise Exception(
#                    "Unexpected error, please make sure paddlehub >= 1.6.2 {}".
#                    format(str(e)))
#        return osp.join(save_dir, fname)
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
            model_type.startswith('DarkNet'):
        for block in program.blocks:
            for param in block.all_parameters():
                pd_var = fluid.global_scope().find_var(param.name)
                pd_param = pd_var.get_tensor()
                if len(np.array(pd_param).shape) == 4:
                    prune_names.append(param.name)
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
        if model_type == 'MobileNetV3_small':
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

    elif model_type.startswith('DeepLabv3p'):
        for param in program.global_block().all_parameters():
            if 'weight' not in param.name:
                continue
            if 'dwise' in param.name or 'depthwise' in param.name or 'logit' in param.name:
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
    else:
        raise Exception('The {} is not implement yet!'.format(model_type))
    return prune_names
