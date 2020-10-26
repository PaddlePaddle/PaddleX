import paddlex
import paddlex.utils.logging as logging
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
    'HRNet_W44':
    'https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W44_C_pretrained.tar',
    'HRNet_W48':
    'https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W48_C_pretrained.tar',
    'HRNet_W60':
    'https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W60_C_pretrained.tar',
    'HRNet_W64':
    'https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W64_C_pretrained.tar',
    'AlexNet':
    'http://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.tar'
}

coco_pretrain = {
    'YOLOv3_DarkNet53_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar',
    'YOLOv3_MobileNetV1_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar',
    'YOLOv3_MobileNetV3_large_COCO':
    'https://bj.bcebos.com/paddlex/models/yolov3_mobilenet_v3.tar',
    'YOLOv3_ResNet34_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar',
    'YOLOv3_ResNet50_vd_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn.tar',
    'FasterRCNN_ResNet18_COCO':
    'https://bj.bcebos.com/paddlex/pretrained_weights/faster_rcnn_r18_fpn_1x.tar',
    'FasterRCNN_ResNet50_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_2x.tar',
    'FasterRCNN_ResNet50_vd_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_2x.tar',
    'FasterRCNN_ResNet101_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_2x.tar',
    'FasterRCNN_ResNet101_vd_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_2x.tar',
    'FasterRCNN_HRNet_W18_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_hrnetv2p_w18_2x.tar',
    'MaskRCNN_ResNet18_COCO':
    'https://bj.bcebos.com/paddlex/pretrained_weights/mask_rcnn_r18_fpn_1x.tar',
    'MaskRCNN_ResNet50_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_2x.tar',
    'MaskRCNN_ResNet50_vd_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar',
    'MaskRCNN_ResNet101_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_fpn_1x.tar',
    'MaskRCNN_ResNet101_vd_COCO':
    'https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_vd_fpn_1x.tar',
    'MaskRCNN_HRNet_W18_COCO':
    'https://bj.bcebos.com/paddlex/pretrained_weights/mask_rcnn_hrnetv2p_w18_2x.tar',
    'UNet_COCO': 'https://paddleseg.bj.bcebos.com/models/unet_coco_v3.tgz',
    'DeepLabv3p_MobileNetV2_x1.0_COCO':
    'https://bj.bcebos.com/v1/paddleseg/deeplab_mobilenet_x1_0_coco.tgz',
    'DeepLabv3p_Xception65_COCO':
    'https://paddleseg.bj.bcebos.com/models/xception65_coco.tgz',
    'PPYOLO_ResNet50_vd_ssld_COCO':
    'https://bj.bcebos.com/paddlex/models/ppyolo_resnet50_vd_ssld.tar'
}

cityscapes_pretrain = {
    'DeepLabv3p_MobileNetV3_large_x1_0_ssld_CITYSCAPES':
    'https://paddleseg.bj.bcebos.com/models/deeplabv3p_mobilenetv3_large_cityscapes.tar.gz',
    'DeepLabv3p_MobileNetV2_x1.0_CITYSCAPES':
    'https://paddleseg.bj.bcebos.com/models/mobilenet_cityscapes.tgz',
    'DeepLabv3p_Xception65_CITYSCAPES':
    'https://paddleseg.bj.bcebos.com/models/xception65_bn_cityscapes.tgz',
    'HRNet_W18_CITYSCAPES':
    'https://paddleseg.bj.bcebos.com/models/hrnet_w18_bn_cityscapes.tgz',
    'FastSCNN_CITYSCAPES':
    'https://paddleseg.bj.bcebos.com/models/fast_scnn_cityscape.tar'
}


def get_pretrain_weights(flag, class_name, backbone, save_dir):
    if flag is None:
        return None
    elif osp.isdir(flag):
        return flag
    elif osp.isfile(flag):
        return flag
    warning_info = "{} does not support to be finetuned with weights pretrained on the {} dataset, so pretrain_weights is forced to be set to {}"
    if flag == 'COCO':
        if class_name == 'DeepLabv3p' and backbone in [
                'Xception41', 'MobileNetV2_x0.25', 'MobileNetV2_x0.5',
                'MobileNetV2_x1.5', 'MobileNetV2_x2.0',
                'MobileNetV3_large_x1_0_ssld'
        ]:
            model_name = '{}_{}'.format(class_name, backbone)
            logging.warning(warning_info.format(model_name, flag, 'IMAGENET'))
            flag = 'IMAGENET'
        elif class_name == 'HRNet':
            logging.warning(warning_info.format(class_name, flag, 'IMAGENET'))
            flag = 'IMAGENET'
        elif class_name == 'FastSCNN':
            logging.warning(warning_info.format(class_name, flag, 'CITYSCAPES'))
            flag = 'CITYSCAPES'
    elif flag == 'CITYSCAPES':
        model_name = '{}_{}'.format(class_name, backbone)
        if class_name == 'UNet':
            logging.warning(warning_info.format(class_name, flag, 'COCO'))
            flag = 'COCO'
        if class_name == 'HRNet' and backbone.split('_')[
                -1] in ['W30', 'W32', 'W40', 'W48', 'W60', 'W64']:
            logging.warning(warning_info.format(backbone, flag, 'IMAGENET'))
            flag = 'IMAGENET'
        if class_name == 'DeepLabv3p' and backbone in [
                'Xception41', 'MobileNetV2_x0.25', 'MobileNetV2_x0.5',
                'MobileNetV2_x1.5', 'MobileNetV2_x2.0'
        ]:
            model_name = '{}_{}'.format(class_name, backbone)
            logging.warning(warning_info.format(model_name, flag, 'IMAGENET'))
            flag = 'IMAGENET'
    elif flag == 'IMAGENET':
        if class_name == 'UNet':
            logging.warning(warning_info.format(class_name, flag, 'COCO'))
            flag = 'COCO'
        elif class_name == 'FastSCNN':
            logging.warning(warning_info.format(class_name, flag, 'CITYSCAPES'))
            flag = 'CITYSCAPES'

    if flag == 'IMAGENET':
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
        if class_name in ['YOLOv3', 'FasterRCNN', 'MaskRCNN']:
            if backbone == 'ResNet50':
                backbone = 'DetResNet50'
        assert backbone in image_pretrain, "There is not ImageNet pretrain weights for {}, you may try COCO.".format(
            backbone)

        if getattr(paddlex, 'gui_mode', False):
            url = image_pretrain[backbone]
            fname = osp.split(url)[-1].split('.')[0]
            paddlex.utils.download_and_decompress(url, path=new_save_dir)
            return osp.join(new_save_dir, fname)

        import paddlehub as hub
        try:
            logging.info(
                "Connecting PaddleHub server to get pretrain weights...")
            hub.download(backbone, save_path=new_save_dir)
        except Exception as e:
            logging.error(
                "Couldn't download pretrain weight, you can download it manualy from {} (decompress the file if it is a compressed file), and set pretrain weights by your self".
                format(image_pretrain[backbone]),
                exit=False)
            if isinstance(e, hub.ResourceNotFoundError):
                raise Exception("Resource for backbone {} not found".format(
                    backbone))
            elif isinstance(e, hub.ServerConnectionError):
                raise Exception(
                    "Cannot get reource for backbone {}, please check your internet connection"
                    .format(backbone))
            else:
                raise Exception(
                    "Unexpected error, please make sure paddlehub >= 1.6.2")
        return osp.join(new_save_dir, backbone)
    elif flag in ['COCO', 'CITYSCAPES']:
        new_save_dir = save_dir
        if hasattr(paddlex, 'pretrain_dir'):
            new_save_dir = paddlex.pretrain_dir
        if class_name in [
                'YOLOv3', 'FasterRCNN', 'MaskRCNN', 'DeepLabv3p', 'PPYOLO'
        ]:
            backbone = '{}_{}'.format(class_name, backbone)
        backbone = "{}_{}".format(backbone, flag)
        if flag == 'COCO':
            url = coco_pretrain[backbone]
        elif flag == 'CITYSCAPES':
            url = cityscapes_pretrain[backbone]
        fname = osp.split(url)[-1].split('.')[0]

        if getattr(paddlex, 'gui_mode', False):
            paddlex.utils.download_and_decompress(url, path=new_save_dir)
            return osp.join(new_save_dir, fname)

        import paddlehub as hub
        try:
            logging.info(
                "Connecting PaddleHub server to get pretrain weights...")
            hub.download(backbone, save_path=new_save_dir)
        except Exception as e:
            logging.error(
                "Couldn't download pretrain weight, you can download it manualy from {} (decompress the file if it is a compressed file), and set pretrain weights by your self".
                format(url),
                exit=False)
            if isinstance(hub.ResourceNotFoundError):
                raise Exception("Resource for backbone {} not found".format(
                    backbone))
            elif isinstance(hub.ServerConnectionError):
                raise Exception(
                    "Cannot get reource for backbone {}, please check your internet connection"
                    .format(backbone))
            else:
                raise Exception(
                    "Unexpected error, please make sure paddlehub >= 1.6.2")
        return osp.join(new_save_dir, backbone)
    else:
        logging.error("Path of retrain weights '{}' is not exists!".format(
            flag))
