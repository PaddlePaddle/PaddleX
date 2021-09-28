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
import time
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


class Times(object):
    def __init__(self):
        self.time = 0.
        # start time
        self.st = 0.
        # end time
        self.et = 0.

    def start(self):
        self.st = time.time()

    def end(self, iter_num=1, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += (self.et - self.st) / iter_num
        else:
            self.time = (self.et - self.st) / iter_num

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 4)


class Timer(Times):
    def __init__(self):
        super(Timer, self).__init__()
        self.preprocess_time_s = Times()
        self.inference_time_s = Times()
        self.postprocess_time_s = Times()
        self.img_num = 0
        self.repeats = 0

    def info(self, average=False):
        total_time = self.preprocess_time_s.value(
        ) * self.img_num + self.inference_time_s.value(
        ) + self.postprocess_time_s.value() * self.img_num
        total_time = round(total_time, 4)
        print("------------------ Inference Time Info ----------------------")
        print("total_time(ms): {}, img_num: {}, batch_size: {}".format(
            total_time * 1000, self.img_num, self.img_num))
        preprocess_time = round(
            self.preprocess_time_s.value() / self.repeats,
            4) if average else self.preprocess_time_s.value()
        postprocess_time = round(
            self.postprocess_time_s.value() / self.repeats,
            4) if average else self.postprocess_time_s.value()
        inference_time = round(self.inference_time_s.value() / self.repeats,
                               4) if average else self.inference_time_s.value()

        average_latency = total_time / self.repeats
        print("average latency time(ms): {:.2f}, QPS: {:2f}".format(
            average_latency * 1000, 1 / average_latency))
        print("preprocess_time_per_im(ms): {:.2f}, "
              "inference_time_per_batch(ms): {:.2f}, "
              "postprocess_time_per_im(ms): {:.2f}".format(
                  preprocess_time * 1000, inference_time * 1000,
                  postprocess_time * 1000))

    def report(self, average=False):
        dic = {}
        dic['preprocess_time_s'] = round(
            self.preprocess_time_s.value() / self.repeats,
            4) if average else self.preprocess_time_s.value()
        dic['postprocess_time_s'] = round(
            self.postprocess_time_s.value() / self.repeats,
            4) if average else self.postprocess_time_s.value()
        dic['inference_time_s'] = round(
            self.inference_time_s.value() / self.repeats,
            4) if average else self.inference_time_s.value()
        dic['img_num'] = self.img_num
        total_time = self.preprocess_time_s.value(
        ) + self.inference_time_s.value() + self.postprocess_time_s.value()
        dic['total_time_s'] = round(total_time, 4)
        dic['batch_size'] = self.img_num / self.repeats
        return dic

    def reset(self):
        self.preprocess_time_s.reset()
        self.inference_time_s.reset()
        self.postprocess_time_s.reset()
        self.img_num = 0
        self.repeats = 0
