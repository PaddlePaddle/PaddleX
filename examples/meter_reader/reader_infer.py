# coding: utf8
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

import os
import os.path as osp
import numpy as np
import math
import cv2
import argparse

from paddlex.seg import transforms
import paddlex as pdx

METER_SHAPE = 512
CIRCLE_CENTER = [256, 256]
CIRCLE_RADIUS = 250
PI = 3.1415926536
LINE_HEIGHT = 120
LINE_WIDTH = 1570
TYPE_THRESHOLD = 40
METER_CONFIG = [{
    'scale_value': 25.0 / 50.0,
    'range': 25.0,
    'unit': "(MPa)"
}, {
    'scale_value': 1.6 / 32.0,
    'range': 1.6,
    'unit': "(MPa)"
}]


def parse_args():
    parser = argparse.ArgumentParser(description='Meter Reader Infering')
    parser.add_argument(
        '--detector_dir',
        dest='detector_dir',
        help='The directory of models to do detection',
        type=str)
    parser.add_argument(
        '--segmenter_dir',
        dest='segmenter_dir',
        help='The directory of models to do segmentation',
        type=str)
    parser.add_argument(
        '--image_dir',
        dest='image_dir',
        help='The directory of images to be infered',
        type=str,
        default=None)
    parser.add_argument(
        '--image',
        dest='image',
        help='The image to be infered',
        type=str,
        default=None)
    parser.add_argument(
        '--use_camera',
        dest='use_camera',
        help='Whether use camera or not',
        action='store_true')
    parser.add_argument(
        '--use_erode',
        dest='use_erode',
        help='Whether erode the predicted lable map',
        action='store_true')
    parser.add_argument(
        '--erode_kernel',
        dest='erode_kernel',
        help='Erode kernel size',
        type=int,
        default=4)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output/result')
    parser.add_argument(
        '--score_threshold',
        dest='score_threshold',
        help="Detected bbox whose score is lower than this threshlod is filtered",
        type=float,
        default=0.5)
    parser.add_argument(
        '--seg_batch_size',
        dest='seg_batch_size',
        help="Segmentation batch size",
        type=int,
        default=2)
    parser.add_argument(
        '--seg_thread_num',
        dest='seg_thread_num',
        help="Thread number of segmentation preprocess",
        type=int,
        default=2)

    return parser.parse_args()


def is_pic(img_name):
    valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
    suffix = img_name.split('.')[-1]
    if suffix not in valid_suffix:
        return False
    return True


class MeterReader:
    def __init__(self, detector_dir, segmenter_dir):
        if not osp.exists(detector_dir):
            raise Exception("Model path {} does not exist".format(
                detector_dir))
        if not osp.exists(segmenter_dir):
            raise Exception("Model path {} does not exist".format(
                segmenter_dir))
        self.detector = pdx.load_model(detector_dir)
        self.segmenter = pdx.load_model(segmenter_dir)
        # Because we will resize images with (METER_SHAPE, METER_SHAPE) before fed into the segmenter,
        # here the transform is composed of normalization only.
        self.seg_transforms = transforms.Compose([transforms.Normalize()])

    def predict(self,
                im_file,
                save_dir='./',
                use_erode=True,
                erode_kernel=4,
                score_threshold=0.5,
                seg_batch_size=2,
                seg_thread_num=2):
        if isinstance(im_file, str):
            im = cv2.imread(im_file).astype('float32')
        else:
            im = im_file.copy()
        # Get detection results
        det_results = self.detector.predict(im)
        # Filter bbox whose score is lower than score_threshold
        filtered_results = list()
        for res in det_results:
            if res['score'] > score_threshold:
                filtered_results.append(res)

        resized_meters = list()
        for res in filtered_results:
            # Crop the bbox area
            xmin, ymin, w, h = res['bbox']
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(im.shape[1], int(xmin + w - 1))
            ymax = min(im.shape[0], int(ymin + h - 1))
            sub_image = im[ymin:(ymax + 1), xmin:(xmax + 1), :]

            # Resize the image with shape (METER_SHAPE, METER_SHAPE)
            meter_shape = sub_image.shape
            scale_x = float(METER_SHAPE) / float(meter_shape[1])
            scale_y = float(METER_SHAPE) / float(meter_shape[0])
            meter_meter = cv2.resize(
                sub_image,
                None,
                None,
                fx=scale_x,
                fy=scale_y,
                interpolation=cv2.INTER_LINEAR)
            meter_meter = meter_meter.astype('float32')
            resized_meters.append(meter_meter)

        meter_num = len(resized_meters)
        seg_results = list()
        for i in range(0, meter_num, seg_batch_size):
            im_size = min(meter_num, i + seg_batch_size)
            meter_images = list()
            for j in range(i, im_size):
                meter_images.append(resized_meters[j - i])
            result = self.segmenter.batch_predict(
                transforms=self.seg_transforms,
                img_file_list=meter_images,
                thread_num=seg_thread_num)
            if use_erode:
                kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
                for i in range(len(seg_results)):
                    results[i]['label_map'] = cv2.erode(
                        seg_results[i]['label_map'], kernel)
            seg_results.extend(result)

        results = list()
        for i, seg_result in enumerate(seg_results):
            result = self.read_process(seg_result['label_map'])
            results.append(result)

        meter_values = list()
        for i, result in enumerate(results):
            if result['scale_num'] > TYPE_THRESHOLD:
                value = result['scales'] * METER_CONFIG[0]['scale_value']
            else:
                value = result['scales'] * METER_CONFIG[1]['scale_value']
            meter_values.append(value)
            print("-- Meter {} -- result: {} --\n".format(i, value))

        # visualize the results
        visual_results = list()
        for i, res in enumerate(filtered_results):
            # Use `score` to represent the meter value
            res['score'] = meter_values[i]
            visual_results.append(res)
        pdx.det.visualize(im_file, visual_results, -1, save_dir=save_dir)

    def read_process(self, label_maps):
        # Convert the circular meter into rectangular meter
        line_images = self.creat_line_image(label_maps)
        # Convert the 2d meter into 1d meter
        scale_data, pointer_data = self.convert_1d_data(line_images)
        # Fliter scale data whose value is lower than the mean value
        self.scale_mean_filtration(scale_data)
        # Get scale_num, scales and ratio of meters
        result = self.get_meter_reader(scale_data, pointer_data)
        return result

    def creat_line_image(self, meter_image):
        line_image = np.zeros((LINE_HEIGHT, LINE_WIDTH), dtype=np.uint8)
        for row in range(LINE_HEIGHT):
            for col in range(LINE_WIDTH):
                theta = PI * 2 / LINE_WIDTH * (col + 1)
                rho = CIRCLE_RADIUS - row - 1
                x = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.5)
                y = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.5)
                line_image[row, col] = meter_image[x, y]
        return line_image

    def convert_1d_data(self, meter_image):
        scale_data = np.zeros((LINE_WIDTH), dtype=np.uint8)
        pointer_data = np.zeros((LINE_WIDTH), dtype=np.uint8)
        for col in range(LINE_WIDTH):
            for row in range(LINE_HEIGHT):
                if meter_image[row, col] == 1:
                    pointer_data[col] += 1
                elif meter_image[row, col] == 2:
                    scale_data[col] += 1
        return scale_data, pointer_data

    def scale_mean_filtration(self, scale_data):
        mean_data = np.mean(scale_data)
        for col in range(LINE_WIDTH):
            if scale_data[col] < mean_data:
                scale_data[col] = 0

    def get_meter_reader(self, scale_data, pointer_data):
        scale_flag = False
        pointer_flag = False
        one_scale_start = 0
        one_scale_end = 0
        one_pointer_start = 0
        one_pointer_end = 0
        scale_location = list()
        pointer_location = 0
        for i in range(LINE_WIDTH - 1):
            if scale_data[i] > 0 and scale_data[i + 1] > 0:
                if scale_flag == False:
                    one_scale_start = i
                    scale_flag = True
            if scale_flag:
                if scale_data[i] == 0 and scale_data[i + 1] == 0:
                    one_scale_end = i - 1
                    one_scale_location = (one_scale_start + one_scale_end) / 2
                    scale_location.append(one_scale_location)
                    one_scale_start = 0
                    one_scale_end = 0
                    scale_flag = False
            if pointer_data[i] > 0 and pointer_data[i + 1] > 0:
                if pointer_flag == False:
                    one_pointer_start = i
                    pointer_flag = True
            if pointer_flag:
                if pointer_data[i] == 0 and pointer_data[i + 1] == 0:
                    one_pointer_end = i - 1
                    pointer_location = (
                        one_pointer_start + one_pointer_end) / 2
                    one_pointer_start = 0
                    one_pointer_end = 0
                    pointer_flag = False

        scale_num = len(scale_location)
        scales = -1
        ratio = -1
        if scale_num > 0:
            for i in range(scale_num - 1):
                if scale_location[
                        i] <= pointer_location and pointer_location < scale_location[
                            i + 1]:
                    scales = i + (pointer_location - scale_location[i]) / (
                        scale_location[i + 1] - scale_location[i] + 1e-05) + 1
            ratio = (pointer_location - scale_location[0]) / (
                scale_location[scale_num - 1] - scale_location[0] + 1e-05)
        result = {'scale_num': scale_num, 'scales': scales, 'ratio': ratio}
        return result


def infer(args):
    image_lists = list()
    if args.image is not None:
        if not osp.exists(args.image):
            raise Exception("Image {} does not exist.".format(args.image))
        if not is_pic(args.image):
            raise Exception("{} is not a picture.".format(args.image))
        image_lists.append(args.image)
    elif args.image_dir is not None:
        if not osp.exists(args.image_dir):
            raise Exception("Directory {} does not exist.".format(
                args.image_dir))
        for im_file in os.listdir(args.image_dir):
            if not is_pic(im_file):
                continue
            im_file = osp.join(args.image_dir, im_file)
            image_lists.append(im_file)

    meter_reader = MeterReader(args.detector_dir, args.segmenter_dir)
    if len(image_lists) > 0:
        for im_file in image_lists:
            meter_reader.predict(im_file, args.save_dir, args.use_erode,
                                 args.erode_kernel, args.score_threshold,
                                 args.seg_batch_size, args.seg_thread_num)
    elif args.with_camera:
        cap_video = cv2.VideoCapture(0)
        if not cap_video.isOpened():
            raise Exception(
                "Error opening video stream, please make sure the camera is working"
            )

        while cap_video.isOpened():
            ret, frame = cap_video.read()
            if ret:
                meter_reader.predict(frame, args.save_dir, args.use_erode,
                                     args.erode_kernel, args.score_threshold,
                                     args.seg_batch_size, args.seg_thread_num)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap_video.release()


if __name__ == '__main__':
    args = parse_args()
    infer(args)
