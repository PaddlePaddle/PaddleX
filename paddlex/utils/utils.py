# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import math
import chardet
import json
import numpy as np
import paddlex
from . import logging
import platform


def seconds_to_hms(seconds):
    h = math.floor(seconds / 3600)
    m = math.floor((seconds - h * 3600) / 60)
    s = int(seconds - h * 3600 - m * 60)
    hms_str = "{}:{}:{}".format(h, m, s)
    return hms_str


def get_encoding(path):
    f = open(path, 'rb')
    data = f.read()
    file_encoding = chardet.detect(data).get('encoding')
    f.close()
    return file_encoding


def get_single_card_bs(batch_size):
    card_num = paddlex.env_info['num']
    place = paddlex.env_info['place']
    if batch_size % card_num == 0:
        return int(batch_size // card_num)
    elif batch_size == 1:
        # Evaluation of detection task only supports single card with batch size 1
        return batch_size
    else:
        raise Exception("Please support correct batch_size, \
                        which can be divided by available cards({}) in {}"
                        .format(card_num, place))


def dict2str(dict_input):
    out = ''
    for k, v in dict_input.items():
        try:
            v = '{:8.6f}'.format(float(v))
        except:
            pass
        out = out + '{}={}, '.format(k, v)
    return out.strip(', ')


def path_normalization(path):
    win_sep = "\\"
    other_sep = "/"
    if platform.system() == "Windows":
        path = win_sep.join(path.split(other_sep))
    else:
        path = other_sep.join(path.split(win_sep))
    return path


def is_pic(img_name):
    valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
    suffix = img_name.split('.')[-1]
    if suffix not in valid_suffix:
        return False
    return True


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class EarlyStop:
    def __init__(self, patience, thresh):
        self.patience = patience
        self.counter = 0
        self.score = None
        self.max = 0
        self.thresh = thresh
        if patience < 1:
            raise Exception("Argument patience should be a positive integer.")

    def __call__(self, current_score):
        if self.score is None:
            self.score = current_score
            return False
        elif current_score > self.max:
            self.counter = 0
            self.score = current_score
            self.max = current_score
            return False
        else:
            if (abs(self.score - current_score) < self.thresh or
                    current_score < self.score):
                self.counter += 1
                self.score = current_score
                logging.debug("EarlyStopping: %i / %i" %
                              (self.counter, self.patience))
                if self.counter >= self.patience:
                    logging.info("EarlyStopping: Stop training")
                    return True
                return False
            else:
                self.counter = 0
                self.score = current_score
                return False


class DisablePrint(object):
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
