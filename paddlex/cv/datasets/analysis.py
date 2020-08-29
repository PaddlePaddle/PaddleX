# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
import numpy as np
import os.path as osp
import cv2
from PIL import Image
import pickle
import threading
import multiprocessing as mp

import paddlex.utils.logging as logging
from paddlex.utils import path_normalization
from .dataset import get_encoding


class Seg:
    def __init__(self, data_dir, file_list, label_list):
        self.data_dir = data_dir
        self.file_list_path = file_list
        self.file_list = list()
        self.labels = list()
        with open(label_list, encoding=get_encoding(label_list)) as f:
            for line in f:
                item = line.strip()
                self.labels.append(item)

        with open(file_list, encoding=get_encoding(file_list)) as f:
            for line in f:
                if line.count(" ") > 1:
                    raise Exception(
                        "A space is defined as the separator, but it exists in image or label name {}."
                        .format(line))
                items = line.strip().split()
                items[0] = path_normalization(items[0])
                items[1] = path_normalization(items[1])
                full_path_im = osp.join(data_dir, items[0])
                full_path_label = osp.join(data_dir, items[1])
                if not osp.exists(full_path_im):
                    raise IOError('The image file {} is not exist!'.format(
                        full_path_im))
                if not osp.exists(full_path_label):
                    raise IOError('The image file {} is not exist!'.format(
                        full_path_label))
                self.file_list.append([full_path_im, full_path_label])
        self.num_samples = len(self.file_list)

    @staticmethod
    def decode_image(im, label):
        if isinstance(im, np.ndarray):
            if len(im.shape) != 3:
                raise Exception(
                    "im should be 3-dimensions, but now is {}-dimensions".
                    format(len(im.shape)))
        else:
            try:
                im = cv2.imread(im)
            except:
                raise ValueError('Can\'t read The image file {}!'.format(im))
        im = im.astype('float32')
        if label is not None:
            if isinstance(label, np.ndarray):
                if len(label.shape) != 2:
                    raise Exception(
                        "label should be 2-dimensions, but now is {}-dimensions".
                        format(len(label.shape)))

            else:
                try:
                    label = np.asarray(Image.open(label))
                except:
                    ValueError('Can\'t read The label file {}!'.format(label))
        im_height, im_width, _ = im.shape
        label_height, label_width = label.shape
        if im_height != label_height or im_width != label_width:
            raise Exception(
                "The height or width of the image is not same as the label")
        return (im, label)

    def _get_shape(self):
        max_height = max(self.im_height_list)
        max_width = max(self.im_width_list)
        min_height = min(self.im_height_list)
        min_width = min(self.im_width_list)
        shape_info = {
            'max_height': max_height,
            'max_width': max_width,
            'min_height': min_height,
            'min_width': min_width,
        }
        return shape_info

    def _get_label_pixel_info(self):
        pixel_num = np.dot(self.im_height_list, self.im_width_list)
        label_pixel_info = dict()
        for label_value, label_value_num in zip(self.label_value_list,
                                                self.label_value_num_list):
            for v, n in zip(label_value, label_value_num):
                if v not in label_pixel_info.keys():
                    label_pixel_info[v] = [n, float(n) / float(pixel_num)]
                else:
                    label_pixel_info[v][0] += n
                    label_pixel_info[v][1] += float(n) / float(pixel_num)

        return label_pixel_info

    def _get_image_pixel_info(self):
        channel = max([len(im_value) for im_value in self.im_value_list])
        im_pixel_info = [dict() for c in range(channel)]
        for im_value, im_value_num in zip(self.im_value_list,
                                          self.im_value_num_list):
            for c in range(channel):
                for v, n in zip(im_value[c], im_value_num[c]):
                    if v not in im_pixel_info[c].keys():
                        im_pixel_info[c][v] = n
                    else:
                        im_pixel_info[c][v] += n
        mode = osp.split(self.file_list_path)[-1].split('.')[0]
        with open(
                osp.join(self.data_dir,
                         '{}_image_pixel_info.pkl'.format(mode)), 'wb') as f:
            pickle.dump(im_pixel_info, f)

        import matplotlib.pyplot as plt
        plot_id = (channel // 3 + 1) * 100 + 31
        for c in range(channel):
            if c > 8:
                continue
            plt.subplot(plot_id + c)
            plt.bar(im_pixel_info[c].keys(),
                    im_pixel_info[c].values(),
                    width=1,
                    log=True)
            plt.xlabel('image pixel value')
            plt.ylabel('number')
            plt.title('channel={}'.format(c))
        plt.savefig(
            osp.join(self.data_dir, '{}_image_pixel_info.png'.format(mode)),
            dpi=800)
        plt.close()
        return im_pixel_info

    def _get_mean_std(self):
        im_mean = np.asarray(self.im_mean_list)
        im_mean = im_mean.sum(axis=0)
        im_mean = im_mean / len(self.file_list)
        im_mean /= 255.

        im_std = np.asarray(self.im_std_list)
        im_std = im_std.sum(axis=0)
        im_std = im_std / len(self.file_list)
        im_std /= 255.

        return (im_mean, im_std)

    def _get_image_info(self, start, end):
        for id in range(start, end):
            full_path_im, full_path_label = self.file_list[id]
            image, label = self.decode_image(full_path_im, full_path_label)

            height, width, channel = image.shape
            self.im_height_list[id] = height
            self.im_width_list[id] = width
            self.im_channel_list[id] = channel

            self.im_mean_list[
                id] = [np.mean(image[:, :, c]) for c in range(channel)]
            self.im_std_list[
                id] = [np.mean(image[:, :, c]) for c in range(channel)]
            for c in range(channel):
                unique, counts = np.unique(image[:, :, c], return_counts=True)
                self.im_value_list[id].extend([unique])
                self.im_value_num_list[id].extend([counts])

            unique, counts = np.unique(label, return_counts=True)
            self.label_value_list[id] = unique
            self.label_value_num_list[id] = counts

    def _get_clipped_mean_std(self, start, end, clip_min_value,
                              clip_max_value):
        for id in range(start, end):
            full_path_im, full_path_label = self.file_list[id]
            image, label = self.decode_image(full_path_im, full_path_label)
            for c in range(self.channel_num):
                np.clip(
                    image[:, :, c],
                    clip_min_value[c],
                    clip_max_value[c],
                    out=image[:, :, c])
                image[:, :, c] -= clip_min_value[c]
                image[:, :, c] /= clip_max_value[c] - clip_min_value[c]
            self.clipped_im_mean_list[id] = [
                image[:, :, c].mean() for c in range(self.channel_num)
            ]
            self.clipped_im_std_list[
                id] = [image[:, :, c].std() for c in range(self.channel_num)]

    def analysis(self):
        self.im_mean_list = [[] for i in range(len(self.file_list))]
        self.im_std_list = [[] for i in range(len(self.file_list))]
        self.im_value_list = [[] for i in range(len(self.file_list))]
        self.im_value_num_list = [[] for i in range(len(self.file_list))]
        self.im_height_list = np.zeros(len(self.file_list), dtype='int32')
        self.im_width_list = np.zeros(len(self.file_list), dtype='int32')
        self.im_channel_list = np.zeros(len(self.file_list), dtype='int32')
        self.label_value_list = [[] for i in range(len(self.file_list))]
        self.label_value_num_list = [[] for i in range(len(self.file_list))]

        num_workers = mp.cpu_count() // 2 if mp.cpu_count() // 2 < 8 else 8
        num_workers = 6
        threads = []
        one_worker_file = len(self.file_list) // num_workers
        for i in range(num_workers):
            start = one_worker_file * i
            end = one_worker_file * (
                i + 1) if i < num_workers - 1 else len(self.file_list)
            t = threading.Thread(
                target=self._get_image_info, args=(start, end))
            print("====", len(self.file_list), start, end)
            #t.daemon = True
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print('ok')
        import time
        import sys
        sys.exit(0)
        time.sleep(1000000)
        return

        #self._get_image_info(0, len(self.file_list))
        unique, counts = np.unique(self.im_channel_list, return_counts=True)
        print('==== unique')
        if len(unique) > 1:
            raise Exception("There are {} kinds of image channels: {}.".format(
                len(unique), unique[:]))
        self.channel_num = unique[0]
        shape_info = self._get_shape()
        print('==== shape_info')
        self.max_height = shape_info['max_height']
        self.max_width = shape_info['max_width']
        self.min_height = shape_info['min_height']
        self.min_width = shape_info['min_width']
        self.label_pixel_info = self._get_label_pixel_info()
        print('==== label_pixel_info')
        self.im_pixel_info = self._get_image_pixel_info()
        print('==== im_pixel_info')
        im_mean, im_std = self._get_mean_std()
        print('==== get_mean_std')
        max_im_value = list()
        min_im_value = list()
        for c in range(self.channel_num):
            max_im_value.append(max(self.im_pixel_info[c].keys()))
            min_im_value.append(min(self.im_pixel_info[c].keys()))
        self.max_im_value = np.asarray(max_im_value)
        self.min_im_value = np.asarray(min_im_value)

        logging.info(
            "############## The analysis results are as follows ##############\n"
        )
        logging.info("{} samples in file {}\n".format(
            len(self.file_list), self.file_list_path))
        logging.info("Maximal image height: {} Maximal image width: {}.\n".
                     format(self.max_height, self.max_width))
        logging.info("Minimal image height: {} Minimal image width: {}.\n".
                     format(self.min_height, self.min_width))
        logging.info("Image channel is {}.\n".format(self.channel_num))
        logging.info(
            "Image mean value: {} Image standard deviation: {} (normalized by 255, sorted by a BGR format).\n".
            format(im_mean, im_std))
        logging.info(
            "Label pixel information is shown in a format of (label_id, the number of label_id, the ratio of label_id):"
        )
        for v, (n, r) in self.label_pixel_info.items():
            logging.info("({}, {}, {})".format(v, n, r))
        mode = osp.split(self.file_list_path)[-1].split('.')[0]
        saved_pkl_file = osp.join(self.data_dir,
                                  '{}_image_pixel_info.pkl'.format(mode))
        saved_png_file = osp.join(self.data_dir,
                                  '{}_image_pixel_info.png'.format(mode))
        logging.info(
            "Image pixel information is saved in the file '{}' and shown in the file '{}'".
            format(saved_pkl_file, saved_png_file))

    def cal_clipvalue_ratio(self, clip_min_value, clip_max_value):
        if len(clip_min_value) != self.channel_num or len(
                clip_max_value) != self.channel_num:
            raise Exception(
                "The length of clip_min_value or clip_max_value should be equal to the number of image channel {}."
                .format(self.channle_num))
        for c in range(self.channel_num):
            if clip_min_value[c] < self.min_im_value[c] or clip_min_value[
                    c] > self.max_im_value[c]:
                raise Exception(
                    "Clip_min_value of the channel {} is not in [{}, {}]".
                    format(c, self.min_im_value[c], self.max_im_value[c]))
            if clip_max_value[c] < self.min_im_value[c] or clip_max_value[
                    c] > self.max_im_value[c]:
                raise Exception(
                    "Clip_max_value of the channel {} is not in [{}, {}]".
                    format(c, self.min_im_value[c], self.max_im_value[c]))
            clip_pixel_num = 0
            pixel_num = sum(self.im_pixel_info[c].values())
            for v, n in self.im_pixel_info[c].items():
                if v < clip_min_value[c] or v > clip_max_value[c]:
                    clip_pixel_num += n
            logging.info("Channel {}, the ratio of pixels to be clipped = {}".
                         format(c, clip_pixel_num / pixel_num))

    def cal_clipped_mean_std(self, clip_min_value, clip_max_value):
        for c in range(self.channel_num):
            if clip_min_value[c] < self.min_im_value[c] or clip_min_value[
                    c] > self.max_im_value[c]:
                raise Exception(
                    "Clip_min_value of the channel {} is not in [{}, {}]".
                    format(c, self.min_im_value[c], self.max_im_value[c]))
            if clip_max_value[c] < self.min_im_value[c] or clip_max_value[
                    c] > self.max_im_value[c]:
                raise Exception(
                    "Clip_max_value of the channel {} is not in [{}, {}]".
                    format(c, self.min_im_value[c], self.max_im_value[c]))

        self.clipped_im_mean_list = [[] for i in range(len(self.file_list))]
        self.clipped_im_std_list = [[] for i in range(len(self.file_list))]

        num_workers = mp.cpu_count() // 2 if mp.cpu_count() // 2 < 8 else 8
        threads = []
        one_worker_file = len(self.file_list) // num_workers
        for i in range(num_workers):
            start = one_worker_file * i
            end = one_worker_file * (
                i + 1) if i < num_workers - 1 else len(self.file_list)
            t = threading.Thread(
                target=self._get_clipped_mean_std,
                args=(start, end, clip_min_value, clip_max_value))
            threads.append(t)
        for t in threads:
            t.setDaemon(True)
            t.start()
        t.join()

        im_mean = np.asarray(self.clipped_im_mean_list)
        im_mean = im_mean.sum(axis=0)
        im_mean = im_mean / len(self.file_list)

        im_std = np.asarray(self.clipped_im_std_list)
        im_std = im_std.sum(axis=0)
        im_std = im_std / len(self.file_list)

        logging.info(
            "Image mean value: {} Image standard deviation: {} (normalized by (clip_max_value - clip_min_value)).\n".
            format(im_mean, im_std))
