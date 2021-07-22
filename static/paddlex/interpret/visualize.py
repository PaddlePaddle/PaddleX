# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import copy
import os.path as osp
import numpy as np
import paddlex as pdx
from .interpretation_predict import interpretation_predict
from .core.interpretation import Interpretation
from .core.normlime_base import precompute_global_classifier
from .core._session_preparation import gen_user_home
from paddlex.cv.transforms import arrange_transforms


def lime(img_file, model, num_samples=3000, batch_size=50, save_dir='./'):
    """使用LIME算法将模型预测结果的可解释性可视化。

    LIME表示与模型无关的局部可解释性，可以解释任何模型。LIME的思想是以输入样本为中心，
    在其附近的空间中进行随机采样，每个采样通过原模型得到新的输出，这样得到一系列的输入
    和对应的输出，LIME用一个简单的、可解释的模型（比如线性回归模型）来拟合这个映射关系，
    得到每个输入维度的权重，以此来解释模型。

    注意：LIME可解释性结果可视化目前只支持分类模型。

    Args:
        img_file (str): 预测图像路径。
        model (paddlex.cv.models): paddlex中的模型。
        num_samples (int): LIME用于学习线性模型的采样数，默认为3000。
        batch_size (int): 预测数据batch大小，默认为50。
        save_dir (str): 可解释性可视化结果（保存为png格式文件）和中间文件存储路径。
    """
    assert model.model_type == 'classifier', \
        'Now the interpretation visualize only be supported in classifier!'
    if model.status != 'Normal':
        raise Exception(
            'The interpretation only can deal with the Normal model')
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    arrange_transforms(
        model.model_type,
        model.__class__.__name__,
        transforms=model.test_transforms,
        mode='test')
    tmp_transforms = copy.deepcopy(model.test_transforms)
    tmp_transforms.transforms = tmp_transforms.transforms[:-2]
    img = tmp_transforms(img_file)[0]
    img = np.around(img).astype('uint8')
    img = np.expand_dims(img, axis=0)
    interpreter = None
    interpreter = get_lime_interpreter(
        img, model, num_samples=num_samples, batch_size=batch_size)
    img_name = osp.splitext(osp.split(img_file)[-1])[0]
    interpreter.interpret(img, save_dir=osp.join(save_dir, img_name))


def normlime(img_file,
             model,
             dataset=None,
             num_samples=3000,
             batch_size=50,
             save_dir='./',
             normlime_weights_file=None):
    """使用NormLIME算法将模型预测结果的可解释性可视化。

    NormLIME是利用一定数量的样本来出一个全局的解释。由于NormLIME计算量较大，此处采用一种简化的方式：
    使用一定数量的测试样本（目前默认使用所有测试样本），对每个样本进行特征提取，映射到同一个特征空间；
    然后以此特征做为输入，以模型输出做为输出，使用线性回归对其进行拟合，得到一个全局的输入和输出的关系。
    之后，对一测试样本进行解释时，使用NormLIME全局的解释，来对LIME的结果进行滤波，使最终的可视化结果更加稳定。

    注意1：dataset读取的是一个数据集，该数据集不宜过大，否则计算时间会较长，但应包含所有类别的数据。
    注意2：NormLIME可解释性结果可视化目前只支持分类模型。

    Args:
        img_file (str): 预测图像路径。
        model (paddlex.cv.models): paddlex中的模型。
        dataset (paddlex.datasets): 数据集读取器，默认为None。
        num_samples (int): LIME用于学习线性模型的采样数，默认为3000。
        batch_size (int): 预测数据batch大小，默认为50。
        save_dir (str): 可解释性可视化结果（保存为png格式文件）和中间文件存储路径。
        normlime_weights_file (str): NormLIME初始化文件名，若不存在，则计算一次，保存于该路径；若存在，则直接载入。
    """
    assert model.model_type == 'classifier', \
        'Now the interpretation visualize only be supported in classifier!'
    if model.status != 'Normal':
        raise Exception(
            'The interpretation only can deal with the Normal model')
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    arrange_transforms(
        model.model_type,
        model.__class__.__name__,
        transforms=model.test_transforms,
        mode='test')
    tmp_transforms = copy.deepcopy(model.test_transforms)
    tmp_transforms.transforms = tmp_transforms.transforms[:-2]
    img = tmp_transforms(img_file)[0]
    img = np.around(img).astype('uint8')
    img = np.expand_dims(img, axis=0)
    interpreter = None
    if dataset is None:
        raise Exception(
            'The dataset is None. Cannot implement this kind of interpretation')
    interpreter = get_normlime_interpreter(
        img,
        model,
        dataset,
        num_samples=num_samples,
        batch_size=batch_size,
        save_dir=save_dir,
        normlime_weights_file=normlime_weights_file)
    img_name = osp.splitext(osp.split(img_file)[-1])[0]
    interpreter.interpret(img, save_dir=osp.join(save_dir, img_name))


def get_lime_interpreter(img, model, num_samples=3000, batch_size=50):
    def predict_func(image):
        out = interpretation_predict(model, image)
        return out[0]

    labels_name = None
    if hasattr(model, 'labels'):
        labels_name = model.labels
    interpreter = Interpretation(
        'lime',
        predict_func,
        labels_name,
        num_samples=num_samples,
        batch_size=batch_size)
    return interpreter


def get_normlime_interpreter(img,
                             model,
                             dataset,
                             num_samples=3000,
                             batch_size=50,
                             save_dir='./',
                             normlime_weights_file=None):
    def predict_func(image):
        out = interpretation_predict(model, image)
        return out[0]

    labels_name = None
    if dataset is not None:
        labels_name = dataset.labels
    root_path = gen_user_home()
    root_path = osp.join(root_path, '.paddlex')
    pre_models_path = osp.join(root_path, "pre_models")
    if not osp.exists(pre_models_path):
        if not osp.exists(root_path):
            os.makedirs(root_path)
        url = "https://bj.bcebos.com/paddlex/interpret/pre_models.tar.gz"
        pdx.utils.download_and_decompress(url, path=root_path)

    if osp.exists(osp.join(save_dir, normlime_weights_file)):
        normlime_weights_file = osp.join(save_dir, normlime_weights_file)
        try:
            np.load(normlime_weights_file, allow_pickle=True).item()
        except:
            normlime_weights_file = precompute_global_classifier(
                dataset,
                predict_func,
                save_path=normlime_weights_file,
                batch_size=batch_size)
    else:
        normlime_weights_file = precompute_global_classifier(
            dataset,
            predict_func,
            save_path=osp.join(save_dir, normlime_weights_file),
            batch_size=batch_size)

    interpreter = Interpretation(
        'normlime',
        predict_func,
        labels_name,
        num_samples=num_samples,
        batch_size=batch_size,
        normlime_weights=normlime_weights_file)
    return interpreter
