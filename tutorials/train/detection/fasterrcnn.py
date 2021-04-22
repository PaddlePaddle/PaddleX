import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx
from paddlex import transforms
import cv2

dataset = 'https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz'
pdx.utils.download_and_decompress(dataset, path='./')

train_transforms = transforms.Compose([
    transforms.RandomResize(
        target_size=[[640, 1333], [672, 1333], [704, 1333], [736, 1333],
                     [768, 1333], [800, 1333]],
        interp=cv2.INTER_CUBIC)
])
