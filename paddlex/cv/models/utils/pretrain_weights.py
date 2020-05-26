import paddlex
import paddlehub as hub
import os
import os.path as osp

image_pretrain = {
    'ResNet18':
    'https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar',
    'ResNet34':
    'https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar',
    'ResNet50':
    'http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar',
    'ResNet101':
    'http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar',
    'ResNet50_vd':
    'https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar',
    'ResNet101_vd':
    'https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar',
    'ResNet50_vd_ssld':
    'https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar',
    'ResNet101_vd_ssld':
    'https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_ssld_pretrained.tar',
    'MobileNetV1':
    'http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar',
    'MobileNetV2_x1.0':
    'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar',
    'MobileNetV2_x0.5':
    'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_5_pretrained.tar',
    'MobileNetV2_x2.0':
    'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x2_0_pretrained.tar',
    'MobileNetV2_x0.25':
    'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_25_pretrained.tar',
    'MobileNetV2_x1.5':
    'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x1_5_pretrained.tar',
    'MobileNetV3_small':
    'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_pretrained.tar',
    'MobileNetV3_large':
    'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_pretrained.tar',
    'MobileNetV3_small_x1_0_ssld':
    'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_ssld_pretrained.tar',
    'MobileNetV3_large_x1_0_ssld':
    'https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_ssld_pretrained.tar',
    'DarkNet53':
    'https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_ImageNet1k_pretrained.tar',
    'DenseNet121':
    'https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet121_pretrained.tar',
    'DenseNet161':
    'https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet161_pretrained.tar',
    'DenseNet201':
    'https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet201_pretrained.tar',
    'DetResNet50':
    'https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar',
    'SegXception41':
    'https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_deeplab_pretrained.tar',
    'SegXception65':
    'https://paddle-imagenet-models-name.bj.bcebos.com/Xception65_deeplab_pretrained.tar',
    'ShuffleNetV2':
    'https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_pretrained.tar',
    'HRNet_W18':
    'https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W18_C_pretrained.tar',
    'HRNet_W30':
    'https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W30_C_pretrained.tar',
    'HRNet_W32':
    'https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W32_C_pretrained.tar',
    'HRNet_W40':
    'https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W40_C_pretrained.tar',
    'HRNet_W48':
    'https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W48_C_pretrained.tar',
    'HRNet_W60':
    'https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W60_C_pretrained.tar',
    'HRNet_W64':
    'https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W64_C_pretrained.tar',
}

coco_pretrain = {
    'UNet': 'https://paddleseg.bj.bcebos.com/models/unet_coco_v3.tgz'
}


def get_pretrain_weights(flag, model_type, backbone, save_dir):
    if flag is None:
        return None
    elif osp.isdir(flag):
        return flag
    elif flag == 'IMAGENET':
        new_save_dir = save_dir
        if hasattr(paddlex, 'pretrain_dir'):
            new_save_dir = paddlex.pretrain_dir
        if backbone.startswith('Xception'):
            backbone = 'Seg{}'.format(backbone)
        elif backbone == 'MobileNetV2':
            backbone = 'MobileNetV2_x1.0'
        elif backbone == 'MobileNetV3_small_ssld':
            backbone = 'MobileNetV3_small_x1_0_ssld'
        elif backbone == 'MobileNetV3_large_ssld':
            backbone = 'MobileNetV3_large_x1_0_ssld'
        if model_type == 'detector':
            if backbone == 'ResNet50':
                backbone = 'DetResNet50'
        assert backbone in image_pretrain, "There is not ImageNet pretrain weights for {}, you may try COCO.".format(
            backbone)
        #        url = image_pretrain[backbone]
        #        fname = osp.split(url)[-1].split('.')[0]
        #        paddlex.utils.download_and_decompress(url, path=new_save_dir)
        #        return osp.join(new_save_dir, fname)
        try:
            hub.download(backbone, save_path=new_save_dir)
        except Exception as e:
            if isinstance(e, hub.ResourceNotFoundError):
                raise Exception("Resource for backbone {} not found".format(
                    backbone))
            elif isinstance(e, hub.ServerConnectionError):
                raise Exception(
                    "Cannot get reource for backbone {}, please check your internet connecgtion"
                    .format(backbone))
            else:
                raise Exception(
                    "Unexpected error, please make sure paddlehub >= 1.6.2")
        return osp.join(new_save_dir, backbone)
    elif flag == 'COCO':
        new_save_dir = save_dir
        if hasattr(paddlex, 'pretrain_dir'):
            new_save_dir = paddlex.pretrain_dir
        url = coco_pretrain[backbone]
        fname = osp.split(url)[-1].split('.')[0]
        #        paddlex.utils.download_and_decompress(url, path=new_save_dir)
        #        return osp.join(new_save_dir, fname)

        assert backbone in coco_pretrain, "There is not COCO pretrain weights for {}, you may try ImageNet.".format(
            backbone)
        try:
            hub.download(backbone, save_path=new_save_dir)
        except Exception as e:
            if isinstance(hub.ResourceNotFoundError):
                raise Exception("Resource for backbone {} not found".format(
                    backbone))
            elif isinstance(hub.ServerConnectionError):
                raise Exception(
                    "Cannot get reource for backbone {}, please check your internet connecgtion"
                    .format(backbone))
            else:
                raise Exception(
                    "Unexpected error, please make sure paddlehub >= 1.6.2")
        return osp.join(new_save_dir, backbone)
    else:
        raise Exception(
            "pretrain_weights need to be defined as directory path or `IMAGENET` or 'COCO' (download pretrain weights automatically)."
        )
