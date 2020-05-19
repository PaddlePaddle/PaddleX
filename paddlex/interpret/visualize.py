#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import cv2
import copy
import os.path as osp
import numpy as np
import paddlex as pdx
from .interpretation_predict import interpretation_predict
from .core.interpretation import Interpretation
from .core.normlime_base import precompute_normlime_weights


def visualize(img_file, 
              model, 
              dataset=None,
              algo='lime',
              num_samples=3000, 
              batch_size=50,
              save_dir='./'):
    """可解释性可视化。
    Args:
        img_file (str): 预测图像路径。
        model (paddlex.cv.models): paddlex中的模型。
        dataset (paddlex.datasets): 数据集读取器，默认为None。
        algo (str): 可解释性方式，当前可选'lime'和'normlime'。
        num_samples (int): LIME用于学习线性模型的采样数，默认为3000。
        batch_size (int): 预测数据batch大小，默认为50。
        save_dir (str): 可解释性可视化结果（保存为png格式文件）和中间文件存储路径。        
    """
    assert model.model_type == 'classifier', \
        'Now the interpretation visualize only be supported in classifier!'
    if model.status != 'Normal':
        raise Exception('The interpretation only can deal with the Normal model')
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    model.arrange_transforms(
                transforms=model.test_transforms, mode='test')
    tmp_transforms = copy.deepcopy(model.test_transforms)
    tmp_transforms.transforms = tmp_transforms.transforms[:-2]
    img = tmp_transforms(img_file)[0]
    img = np.around(img).astype('uint8')
    img = np.expand_dims(img, axis=0)
    interpreter = None
    if algo == 'lime':
        interpreter = get_lime_interpreter(img, model, dataset, num_samples=num_samples, batch_size=batch_size)
    elif algo == 'normlime':
        if dataset is None:
            raise Exception('The dataset is None. Cannot implement this kind of interpretation')
        interpreter = get_normlime_interpreter(img, model, dataset, 
                                     num_samples=num_samples, batch_size=batch_size,
                                     save_dir=save_dir)
    else:
        raise Exception('The {} interpretation method is not supported yet!'.format(algo))
    img_name = osp.splitext(osp.split(img_file)[-1])[0]
    interpreter.interpret(img, save_dir=save_dir)
    
    
def get_lime_interpreter(img, model, dataset, num_samples=3000, batch_size=50):
    def predict_func(image):
        image = image.astype('float32')
        for i in range(image.shape[0]):
            image[i] = cv2.cvtColor(image[i], cv2.COLOR_RGB2BGR)
        tmp_transforms = copy.deepcopy(model.test_transforms.transforms)
        model.test_transforms.transforms = model.test_transforms.transforms[-2:]
        out = interpretation_predict(model, image)
        model.test_transforms.transforms = tmp_transforms
        return out[0]
    labels_name = None
    if dataset is not None:
        labels_name = dataset.labels
    interpreter = Interpretation('lime', 
                            predict_func,
                            labels_name,
                            num_samples=num_samples, 
                            batch_size=batch_size)
    return interpreter


def get_normlime_interpreter(img, model, dataset, num_samples=3000, batch_size=50, save_dir='./'):
    def precompute_predict_func(image):
        image = image.astype('float32')
        tmp_transforms = copy.deepcopy(model.test_transforms.transforms)
        model.test_transforms.transforms = model.test_transforms.transforms[-2:]
        out = interpretation_predict(model, image)
        model.test_transforms.transforms = tmp_transforms
        return out[0]
    def predict_func(image):
        image = image.astype('float32')
        for i in range(image.shape[0]):
            image[i] = cv2.cvtColor(image[i], cv2.COLOR_RGB2BGR)
        tmp_transforms = copy.deepcopy(model.test_transforms.transforms)
        model.test_transforms.transforms = model.test_transforms.transforms[-2:]
        out = interpretation_predict(model, image)
        model.test_transforms.transforms = tmp_transforms
        return out[0]
    labels_name = None
    if dataset is not None:
        labels_name = dataset.labels
    root_path = os.environ['HOME']
    root_path = osp.join(root_path, '.paddlex')
    pre_models_path = osp.join(root_path, "pre_models")
    if not osp.exists(pre_models_path):
        os.makedirs(root_path)
        url = "https://bj.bcebos.com/paddlex/interpret/pre_models.tar.gz"
        pdx.utils.download_and_decompress(url, path=root_path)
    npy_dir = precompute_for_normlime(precompute_predict_func, 
                                      dataset, 
                                      num_samples=num_samples, 
                                      batch_size=batch_size,
                                      save_dir=save_dir)
    interpreter = Interpretation('normlime', 
                            predict_func,
                            labels_name,
                            num_samples=num_samples, 
                            batch_size=batch_size,
                            normlime_weights=npy_dir)
    return interpreter


def precompute_for_normlime(predict_func, dataset, num_samples=3000, batch_size=50, save_dir='./'):
    image_list = []
    for item in dataset.file_list:
        image_list.append(item[0])
    return precompute_normlime_weights(
            image_list,  
            predict_func,
            num_samples=num_samples, 
            batch_size=batch_size,
            save_dir=save_dir)
  
