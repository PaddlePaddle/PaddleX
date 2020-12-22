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


def get_base_params(model_type, per_gpu_memory, num_train_samples, num_gpu,
                    gpu_list, num_classes):
    params = dict()
    params["model"] = model_type
    params["cpu_num"] = 1
    if num_gpu == 0:
        batch_size = 4
        params['cuda_visible_devices'] = ''
    else:
        params['cuda_visible_devices'] = str(gpu_list).strip("[]")
        if model_type.startswith('MobileNet'):
            batch_size = (per_gpu_memory - 513) // 57 * num_gpu
            batch_size = min(batch_size, num_gpu * 125)
        elif model_type.startswith('DenseNet') or model_type.startswith('ResNet') \
            or model_type.startswith('Xception') or model_type.startswith('DarkNet') \
            or model_type.startswith('ShuffleNet'):
            batch_size = (per_gpu_memory - 739) // 211 * num_gpu
            batch_size = min(batch_size, num_gpu * 16)
        elif model_type.startswith('YOLOv3'):
            batch_size = (per_gpu_memory - 1555) // 943 * num_gpu
            batch_size = min(batch_size, num_gpu * 8)
        elif model_type.startswith('PPYOLO'):
            batch_size = (per_gpu_memory - 1691) // 1025 * num_gpu
            batch_size = min(batch_size, num_gpu * 8)
        elif model_type.startswith('FasterRCNN'):
            batch_size = (per_gpu_memory - 1755) // 915 * num_gpu
            batch_size = min(batch_size, num_gpu * 2)
        elif model_type.startswith('MaskRCNN'):
            batch_size = (per_gpu_memory - 2702) // 1188 * num_gpu
            batch_size = min(batch_size, num_gpu * 2)
        elif model_type.startswith('DeepLab'):
            batch_size = (per_gpu_memory - 1469) // 1605 * num_gpu
            batch_size = min(batch_size, num_gpu * 4)
        elif model_type.startswith('UNet'):
            batch_size = (per_gpu_memory - 1275) // 1256 * num_gpu
            batch_size = min(batch_size, num_gpu * 4)
        elif model_type.startswith('HRNet_W18'):
            batch_size = (per_gpu_memory - 800) // 682 * num_gpu
            batch_size = min(batch_size, num_gpu * 4)
        elif model_type.startswith('FastSCNN'):
            batch_size = (per_gpu_memory - 636) // 144 * num_gpu
            batch_size = min(batch_size, num_gpu * 4)
    if batch_size > num_train_samples // 2:
        batch_size = num_train_samples // 2
    if batch_size < 1:
        batch_size = 1

    brightness_range = 0.5
    contrast_range = 0.5
    saturation_range = 0.5
    saturation = False
    hue = False
    if model_type.startswith('DenseNet') or model_type.startswith('ResNet') \
        or model_type.startswith('Xception') or model_type.startswith('DarkNet') \
        or model_type.startswith('ShuffleNet') or model_type.startswith('MobileNet'):
        if model_type.startswith('MobileNet'):
            lr = (batch_size / 500.0) * 0.1
        else:
            lr = (batch_size / 256.0) * 0.1
        shape = [224, 224]
        save_interval_epochs = 5
        num_epochs = 120
        lr_decay_epochs = [30, 60, 90]
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        brightness_range = 0.9
        contrast_range = 0.9
        saturation_range = 0.9
        brightness = True
        contrast = True
    elif model_type.startswith('YOLOv3') or model_type.startswith('PPYOLO'):
        shape = [608, 608]
        save_interval_epochs = 30
        num_epochs = 270
        lr_decay_epochs = [210, 240]
        lr = 0.001 * batch_size / 64
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        brightness = True
        contrast = True
        saturation = True
        hue = True
        num_steps_each_epoch = num_train_samples // batch_size
        min_warmup_step = max(3 * num_steps_each_epoch, 50 * num_classes)
        if num_gpu == 0:
            num_gpu = 1
        warmup_step = min(min_warmup_step, int(400 * num_classes / num_gpu))
    elif model_type.startswith('FasterRCNN') or model_type.startswith(
            'MaskRCNN'):
        shape = [800, 1333]
        save_interval_epochs = 1
        num_epochs = 12
        lr_decay_epochs = [8, 11]
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        brightness = False
        contrast = False
        lr = 0.02 * batch_size / 16
        num_steps_each_epoch = num_train_samples // batch_size
        min_warmup_step = max(num_steps_each_epoch, 50)
        if num_gpu == 0:
            num_gpu = 1
        warmup_step = min(min_warmup_step, int(4000 / num_gpu))
    elif model_type.startswith('DeepLab') or model_type.startswith('UNet') \
        or model_type.startswith('HRNet_W18') or model_type.startswith('FastSCNN'):
        shape = [512, 512]
        save_interval_epochs = 10
        num_epochs = 100
        lr_decay_epochs = [10, 20]
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]
        brightness = False
        contrast = False
        lr = 0.01 * batch_size / 2

    params['batch_size'] = batch_size
    params['learning_rate'] = lr
    params["image_shape"] = shape
    params['save_interval_epochs'] = save_interval_epochs
    params['num_epochs'] = num_epochs
    params['lr_decay_epochs'] = lr_decay_epochs
    params['resume_checkpoint'] = None
    params["sensitivities_path"] = None
    params["image_mean"] = image_mean
    params["image_std"] = image_std
    params["horizontal_flip_prob"] = 0.5
    params['brightness'] = brightness
    params["brightness_range"] = brightness_range
    params["brightness_prob"] = 0.5
    params['contrast'] = contrast
    params['contrast_range'] = contrast_range
    params['contrast_prob'] = 0.5
    params['saturation'] = saturation
    params['saturation_range'] = saturation_range
    params['saturation_prob'] = 0.5
    params['hue'] = hue
    params['hue_range'] = 18
    params['hue_prob'] = 0.5
    params['horizontal_flip'] = True

    if model_type in ['YOLOv3', 'PPYOLO', 'FasterRCNN', 'MaskRCNN']:
        num_epochs = params['num_epochs']
        lr_decay_epochs = params['lr_decay_epochs']
        if warmup_step >= lr_decay_epochs[0] * num_steps_each_epoch:
            for i in range(len(lr_decay_epochs)):
                lr_decay_epochs[i] += warmup_step // num_steps_each_epoch
            num_epochs += warmup_step // num_steps_each_epoch
        params['num_epochs'] = num_epochs
        params['lr_decay_epochs'] = lr_decay_epochs
        params['warmup_steps'] = warmup_step

    return params


def get_classification_params(params):
    params["pretrain_weights"] = 'IMAGENET'
    params["lr_policy"] = "Piecewise"
    params['vertical_flip_prob'] = 0.5
    params['vertical_flip'] = True
    params['rotate'] = True
    params['rotate_prob'] = 0.5
    params['rotate_range'] = 30
    return params


def get_detection_params(params):
    params['with_fpn'] = True
    params["pretrain_weights"] = 'IMAGENET'
    if params['model'].startswith('YOLOv3') or params['model'].startswith(
            'PPYOLO'):
        if params['model'].startswith('YOLOv3'):
            params['backbone'] = 'DarkNet53'
        elif params['model'].startswith('PPYOLO'):
            params['backbone'] = 'ResNet50_vd_ssld'
        params['use_mixup'] = True
        params['mixup_alpha'] = 1.5
        params['mixup_beta'] = 1.5
        params['expand_prob'] = 0.5
        params['expand_image'] = True
        params['crop_image'] = True
        params['random_shape'] = True
        params['random_shape_sizes'] = [
            320, 352, 384, 416, 448, 480, 512, 544, 576, 608
        ]
    elif params['model'].startswith('FasterRCNN') or params[
            'model'].startswith('MaskRCNN'):
        params['backbone'] = 'ResNet50'

    return params


def get_segmentation_params(params):
    if params['model'].startswith('DeepLab'):
        params['backbone'] = 'Xception65'
        params["pretrain_weights"] = 'IMAGENET'
    elif params['model'].startswith('UNet'):
        params["pretrain_weights"] = 'COCO'
    elif params['model'].startswith('HRNet_W18') or params['model'].startswith(
            'FastSCNN'):
        params["pretrain_weights"] = 'CITYSCAPES'
    params['loss_type'] = [False, False]
    params['lr_policy'] = 'Polynomial'
    params['optimizer'] = 'SGD'
    params['blur'] = False
    params['blur_prob'] = 0.1
    params['rotate'] = False
    params['max_rotation'] = 15
    params['scale_aspect'] = False
    params['min_ratio'] = 0.5
    params['aspect_ratio'] = 0.33
    params['vertical_flip'] = False
    params['vertical_flip_prob'] = 0.5
    return params


def get_params(data, project_type, num_train_samples, num_classes, num_gpu,
               per_gpu_memory, gpu_list):
    if project_type == "classification":
        if 'model_type' in data:
            model_type = data['model_type']
        else:
            model_type = "MobileNetV2"
        params = get_base_params(model_type, per_gpu_memory, num_train_samples,
                                 num_gpu, gpu_list, num_classes)
        return get_classification_params(params)
    if project_type == "detection":
        if 'model_type' in data:
            model_type = data['model_type']
        else:
            model_type = "YOLOv3"
        params = get_base_params(model_type, per_gpu_memory, num_train_samples,
                                 num_gpu, gpu_list, num_classes)
        return get_detection_params(params)
    if project_type == "instance_segmentation":
        if 'model_type' in data:
            model_type = data['model_type']
        else:
            model_type = "MaskRCNN"
        params = get_base_params(model_type, per_gpu_memory, num_train_samples,
                                 num_gpu, gpu_list, num_classes)
        return get_detection_params(params)
    if project_type == 'segmentation' or project_type == "remote_segmentation":
        if 'model_type' in data:
            model_type = data['model_type']
        else:
            model_type = "DeepLabv3+"
        params = get_base_params(model_type, per_gpu_memory, num_train_samples,
                                 num_gpu, gpu_list, num_classes)
        return get_segmentation_params(params)
