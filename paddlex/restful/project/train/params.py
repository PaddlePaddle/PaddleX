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
import platform
import os


class Params(object):
    def __init__(self):
        self.init_train_params()
        self.init_transform_params()

    def init_train_params(self):
        self.cuda_visible_devices = ''
        self.batch_size = 2
        self.save_interval_epochs = 1
        self.pretrain_weights = 'IMAGENET'
        self.model = 'MobileNetV2'
        self.num_epochs = 4
        self.learning_rate = 0.000125
        self.lr_decay_epochs = [2, 3]
        self.train_num = 0
        self.resume_checkpoint = None
        self.sensitivities_path = None
        self.eval_metric_loss = None

    def init_transform_params(self):
        self.image_shape = [224, 224]
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.horizontal_flip_prob = 0.5
        self.brightness_range = 0.9
        self.brightness_prob = 0.5
        self.contrast_range = 0.9
        self.contrast_prob = 0.5
        self.saturation_range = 0.9
        self.saturation_prob = 0.5
        self.hue_range = 18
        self.hue_prob = 0.5
        self.horizontal_flip = True
        self.brightness = True
        self.contrast = True
        self.saturation = True
        self.hue = True

    def load_from_dict(self, params_dict):
        for attr in params_dict:
            if hasattr(self, attr):
                method = getattr(self, "set_" + attr)
                method(params_dict[attr])

    def set_cuda_visible_devices(self, cuda_visible_devices):
        self.cuda_visible_devices = cuda_visible_devices

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_save_interval_epochs(self, save_interval_epochs):
        self.save_interval_epochs = save_interval_epochs

    def set_pretrain_weights(self, pretrain_weights):
        self.pretrain_weights = pretrain_weights

    def set_model(self, model):
        self.model = model

    def set_num_epochs(self, num_epochs):
        self.num_epochs = num_epochs

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_lr_decay_epochs(self, lr_decay_epochs):
        self.lr_decay_epochs = lr_decay_epochs

    def set_resume_checkpoint(self, resume_checkpoint):
        self.resume_checkpoint = resume_checkpoint

    def set_sensitivities_path(self, sensitivities_path):
        self.sensitivities_path = sensitivities_path

    def set_eval_metric_loss(self, eval_metric_loss):
        self.eval_metric_loss = eval_metric_loss

    def set_image_shape(self, image_shape):
        self.image_shape = image_shape

    def set_image_mean(self, image_mean):
        self.image_mean = image_mean

    def set_image_std(self, image_std):
        self.image_std = image_std

    def set_horizontal_flip(self, horizontal_flip):
        self.horizontal_flip = horizontal_flip
        if not horizontal_flip:
            self.horizontal_flip_prob = 0.0

    def set_horizontal_flip_prob(self, horizontal_flip_prob):
        if self.horizontal_flip:
            self.horizontal_flip_prob = horizontal_flip_prob

    def set_brightness_range(self, brightness_range):
        self.brightness_range = brightness_range

    def set_brightness_prob(self, brightness_prob):
        if self.brightness:
            self.brightness_prob = brightness_prob

    def set_brightness(self, brightness):
        self.brightness = brightness
        if not brightness:
            self.brightness_prob = 0.0

    def set_contrast(self, contrast):
        self.contrast = contrast
        if not contrast:
            self.contrast_prob = 0.0

    def set_contrast_prob(self, contrast_prob):
        if self.contrast:
            self.contrast_prob = contrast_prob

    def set_contrast_range(self, contrast_range):
        self.contrast_range = contrast_range

    def set_saturation(self, saturation):
        self.saturation = saturation
        if not saturation:
            self.saturation_prob = 0.0

    def set_saturation_prob(self, saturation_prob):
        if self.saturation:
            self.saturation_prob = saturation_prob

    def set_saturation_range(self, saturation_range):
        self.saturation_range = saturation_range

    def set_hue(self, hue):
        self.hue = hue
        if not hue:
            self.hue_prob = 0.0

    def set_hue_prob(self, hue_prob):
        if self.hue_prob:
            self.hue_prob = hue_prob

    def set_hue_range(self, hue_range):
        self.hue_range = hue_range

    def set_train_num(self, train_num):
        self.train_num = train_num


class ClsParams(Params):
    def __init__(self):
        super(ClsParams, self).__init__()
        self.lr_policy = 'Piecewise'
        self.vertical_flip_prob = 0.0
        self.vertical_flip = True
        self.rotate_prob = 0.0
        self.rotate_range = 30
        self.rotate = True

    def set_lr_policy(self, lr_policy):
        self.lr_policy = lr_policy

    def set_vertical_flip(self, vertical_flip):
        self.vertical_flip = vertical_flip
        if not self.vertical_flip:
            self.vertical_flip_prob = 0.0

    def set_vertical_flip_prob(self, vertical_flip_prob):
        if self.vertical_flip:
            self.vertical_flip_prob = vertical_flip_prob

    def set_rotate(self, rotate):
        self.rotate = rotate
        if not rotate:
            self.rotate_prob = 0.0

    def set_rotate_prob(self, rotate_prob):
        if self.rotate:
            self.rotate_prob = rotate_prob

    def set_rotate_range(self, rotate_range):
        self.rotate_range = rotate_range


class DetParams(Params):
    def __init__(self):
        super(DetParams, self).__init__()
        self.warmup_steps = 10
        self.warmup_start_lr = 0.
        self.use_mixup = True
        self.mixup_alpha = 1.5
        self.mixup_beta = 1.5
        self.expand_prob = 0.5
        self.expand_image = True
        self.crop_image = True
        self.backbone = 'ResNet18'
        self.model = 'FasterRCNN'
        self.with_fpn = True
        self.random_shape = True
        self.random_shape_sizes = [
            320, 352, 384, 416, 448, 480, 512, 544, 576, 608
        ]

    def set_warmup_steps(self, warmup_steps):
        self.warmup_steps = warmup_steps

    def set_warmup_start_lr(self, warmup_start_lr):
        self.warmup_start_lr = warmup_start_lr

    def set_use_mixup(self, use_mixup):
        self.use_mixup = use_mixup

    def set_mixup_alpha(self, mixup_alpha):
        self.mixup_alpha = mixup_alpha

    def set_mixup_beta(self, mixup_beta):
        self.mixup_beta = mixup_beta

    def set_expand_image(self, expand_image):
        self.expand_image = expand_image
        if not expand_image:
            self.expand_prob = 0.0

    def set_expand_prob(self, expand_prob):
        if self.expand_image:
            self.expand_prob = expand_prob

    def set_crop_image(self, crop_image):
        self.crop_image = crop_image

    def set_backbone(self, backbone):
        self.backbone = backbone

    def set_with_fpn(self, with_fpn):
        self.with_fpn = with_fpn

    def set_random_shape(self, random_shape):
        self.random_shape = random_shape

    def set_random_shape_sizes(self, random_shape_sizes):
        self.random_shape_sizes = random_shape_sizes


class SegParams(Params):
    def __init__(self):
        super(SegParams, self).__init__()
        self.loss_type = [True, True]
        self.lr_policy = 'Piecewise'
        self.optimizer = 'Adam'
        self.backbone = 'MobileNetV2_x1.0'
        self.blur = True
        self.blur_prob = 0.
        self.rotate = False
        self.max_rotation = 15
        self.scale_aspect = False
        self.min_ratio = 0.5
        self.aspect_ratio = 0.33
        self.vertical_flip_prob = 0.0
        self.vertical_flip = True
        self.model = 'UNet'

    def set_loss_type(self, loss_type):
        self.loss_type = loss_type

    def set_lr_policy(self, lr_policy):
        self.lr_policy = lr_policy

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_backbone(self, backbone):
        self.backbone = backbone

    def set_blur(self, blur):
        self.blur = blur
        if not blur:
            self.blur_prob = 0.

    def set_blur_prob(self, blur_prob):
        if self.blur:
            self.blur_prob = blur_prob

    def set_rotate(self, rotate):
        self.rotate = rotate

    def set_max_rotation(self, max_rotation):
        self.max_rotation = max_rotation

    def set_scale_aspect(self, scale_aspect):
        self.scale_aspect = scale_aspect

    def set_min_ratio(self, min_ratio):
        self.min_ratio = min_ratio

    def set_aspect_ratio(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio

    def set_vertical_flip(self, vertical_flip):
        self.vertical_flip = vertical_flip
        if not vertical_flip:
            self.vertical_flip_prob = 0.0

    def set_vertical_flip_prob(self, vertical_flip_prob):
        if vertical_flip_prob:
            self.vertical_flip_prob = vertical_flip_prob


PARAMS_CLASS_LIST = [ClsParams, DetParams, SegParams, DetParams, SegParams]


def recommend_parameters(params, train_nums, class_nums, memory_size_per_gpu):
    model_type = params['model']
    gpu_list = params['cuda_visible_devices']
    if 'cpu_num' in params:
        cpu_num = params['cpu_num']
    else:
        cpu_num = int(os.environ.get('CPU_NUM', 1))
        if cpu_num > 8:
            os.environ['CPU_NUM'] = '8'
    if not params['use_gpu']:
        gpu_nums = 0
    else:
        gpu_nums = len(gpu_list.split(','))

    # set batch_size
    if gpu_nums == 0 or platform.platform().startswith("Darwin"):
        if model_type.startswith('MobileNet'):
            batch_size = 8 * cpu_num
        elif model_type.startswith('DenseNet') or model_type.startswith('ResNet') \
            or model_type.startswith('Xception') or model_type.startswith('DarkNet') \
            or model_type.startswith('ShuffleNet'):
            batch_size = 4 * cpu_num
        elif model_type.startswith('YOLOv3') or model_type.startswith(
                'PPYOLO'):
            batch_size = 2 * cpu_num
        elif model_type.startswith('FasterRCNN') or model_type.startswith(
                'MaskRCNN'):
            batch_size = 1 * cpu_num
        elif model_type.startswith('DeepLab') or model_type.startswith('UNet') \
            or model_type.startswith('HRNet_W18') or model_type.startswith('FastSCNN'):
            batch_size = 2 * cpu_num
    else:
        if model_type.startswith('MobileNet'):
            batch_size = (memory_size_per_gpu - 513) // 57 * gpu_nums
            batch_size = min(batch_size, gpu_nums * 125)
        elif model_type.startswith('DenseNet') or model_type.startswith('ResNet') \
            or model_type.startswith('Xception') or model_type.startswith('DarkNet') \
            or model_type.startswith('ShuffleNet'):
            batch_size = (memory_size_per_gpu - 739) // 211 * gpu_nums
            batch_size = min(batch_size, gpu_nums * 16)
        elif model_type.startswith('YOLOv3'):
            batch_size = (memory_size_per_gpu - 1555) // 943 * gpu_nums
            batch_size = min(batch_size, gpu_nums * 8)
        elif model_type.startswith('PPYOLO'):
            batch_size = (memory_size_per_gpu - 1691) // 1025 * gpu_nums
            batch_size = min(batch_size, gpu_nums * 8)
        elif model_type.startswith('FasterRCNN'):
            batch_size = (memory_size_per_gpu - 1755) // 915 * gpu_nums
            batch_size = min(batch_size, gpu_nums * 2)
        elif model_type.startswith('MaskRCNN'):
            batch_size = (memory_size_per_gpu - 2702) // 1188 * gpu_nums
            batch_size = min(batch_size, gpu_nums * 2)
        elif model_type.startswith('DeepLab'):
            batch_size = (memory_size_per_gpu - 1469) // 1605 * gpu_nums
            batch_size = min(batch_size, gpu_nums * 4)
        elif model_type.startswith('UNet'):
            batch_size = (memory_size_per_gpu - 1275) // 1256 * gpu_nums
            batch_size = min(batch_size, gpu_nums * 4)
        elif model_type.startswith('HRNet_W18'):
            batch_size = (memory_size_per_gpu - 800) // 682 * gpu_nums
            batch_size = min(batch_size, gpu_nums * 4)
        elif model_type.startswith('FastSCNN'):
            batch_size = (memory_size_per_gpu - 636) // 144 * gpu_nums
            batch_size = min(batch_size, gpu_nums * 4)

    if batch_size > train_nums // 2:
        batch_size = train_nums // 2
        gpu_list = '{}'.format(gpu_list.split(',')[0]) if gpu_nums > 0 else ''
    if batch_size <= 0:
        batch_size = 1

    # set learning_rate
    if model_type.startswith('MobileNet'):
        lr = (batch_size / 500.0) * 0.1
    elif model_type.startswith('DenseNet') or model_type.startswith('ResNet') \
        or model_type.startswith('Xception') or model_type.startswith('DarkNet') \
        or model_type.startswith('ShuffleNet'):
        lr = (batch_size / 256.0) * 0.1
    elif model_type.startswith('YOLOv3') or model_type.startswith('PPYOLO'):
        lr = 0.001 * batch_size / 64
        num_steps_each_epoch = train_nums // batch_size
        min_warmup_step = max(3 * num_steps_each_epoch, 50 * class_nums)
        if gpu_nums == 0:
            gpu_nums = 1
        warmup_step = min(min_warmup_step, int(400 * class_nums / gpu_nums))
    elif model_type.startswith('FasterRCNN') or model_type.startswith(
            'MaskRCNN'):
        lr = 0.02 * batch_size / 16
        num_steps_each_epoch = train_nums // batch_size
        min_warmup_step = max(num_steps_each_epoch, 50)
        if gpu_nums == 0:
            gpu_nums = 1
        warmup_step = min(min_warmup_step, int(4000 / gpu_nums))
    elif model_type.startswith('DeepLab') or model_type.startswith('UNet') \
        or model_type.startswith('HRNet_W18') or model_type.startswith('FastSCNN'):
        lr = 0.01 * batch_size / 2
        loss_type = [False, False]

    params['batch_size'] = batch_size
    params['learning_rate'] = lr
    params['cuda_visible_devices'] = gpu_list
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
    if 'loss_type' in params:
        params['loss_type'] = loss_type
