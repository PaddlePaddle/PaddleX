# coding: utf8
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

import os
import os.path as osp
import numpy as np
import math
import cv2
import argparse

from paddlex import transforms as T
import paddlex as pdx

# 读数后处理中有把圆形表盘转成矩形的操作，矩形的宽即为圆形的外周长
# 因此要求表盘图像大小为固定大小，这里设置为[512, 512]
METER_SHAPE = [512, 512]  # 高x宽
# 圆形表盘的中心点
CIRCLE_CENTER = [256, 256]  # 高x宽
# 圆形表盘的半径
CIRCLE_RADIUS = 250
# 圆周率
PI = 3.1415926536
# 在把圆形表盘转成矩形后矩形的高
# 当前设置值约为半径的一半，原因是：圆形表盘的中心区域除了指针根部就是背景了
# 我们只需要把外围的刻度、指针的尖部保存下来就可以定位出指针指向的刻度
RECTANGLE_HEIGHT = 120
# 矩形表盘的宽，即圆形表盘的外周长
RECTANGLE_WIDTH = 1570
# 当前案例中只使用了两种类型的表盘，第一种表盘的刻度根数为50
# 第二种表盘的刻度根数为32。因此，我们通过预测的刻度根数来判断表盘类型
# 刻度根数超过阈值的即为第一种，否则是第二种
TYPE_THRESHOLD = 40
# 两种表盘的配置信息，包含每根刻度的值，量程，单位
METER_CONFIG = [{
    'scale_interval_value': 25.0 / 50.0,
    'range': 25.0,
    'unit': "(MPa)"
}, {
    'scale_interval_value': 1.6 / 32.0,
    'range': 1.6,
    'unit': "(MPa)"
}]
# 分割模型预测类别id与类别名的对应关系
SEG_CNAME2CLSID = {'background': 0, 'pointer': 1, 'scale': 2}


def parse_args():
    parser = argparse.ArgumentParser(description='Meter Reader Infering')
    parser.add_argument(
        '--det_model_dir',
        dest='det_model_dir',
        help='The directory of the detection model',
        type=str)
    parser.add_argument(
        '--seg_model_dir',
        dest='seg_model_dir',
        help='The directory of the segmentation model',
        type=str)
    parser.add_argument(
        '--image_dir',
        dest='image_dir',
        help='The directory of images to be inferred',
        type=str,
        default=None)
    parser.add_argument(
        '--image',
        dest='image',
        help='The image to be inferred',
        type=str,
        default=None)
    parser.add_argument(
        '--use_erode',
        dest='use_erode',
        help='Whether erode the lable map predicted from a segmentation model',
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
        help='The directory for saving the predicted results',
        type=str,
        default='./output/result')
    parser.add_argument(
        '--score_threshold',
        dest='score_threshold',
        help="Predicted bounding boxes whose scores are lower than this threshlod are filtered",
        type=float,
        default=0.5)
    parser.add_argument(
        '--seg_batch_size',
        dest='seg_batch_size',
        help="The number of images fed into the segmentation model during one forward propagation",
        type=int,
        default=2)

    return parser.parse_args()


def is_pic(img_name):
    """判断是否是图片

    参数：
        img_name (str): 图片路径

    返回：
        flag (bool): 判断值。
    """
    valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
    suffix = img_name.split('.')[-1]
    flag = True
    if suffix not in valid_suffix:
        flag = False
    return flag


class MeterReader:
    """检测表盘的位置，分割各表盘内刻度和指针的位置，并根据分割结果计算出各表盘的读数

    参数：
        det_model_dir (str): 用于定位表盘的检测模型所在路径。
        seg_model_dir (str): 用于分割刻度和指针的分割模型所在路径。

    """

    def __init__(self, det_model_dir, seg_model_dir):
        if not osp.exists(det_model_dir):
            raise Exception("Model path {} does not exist".format(
                det_model_dir))
        if not osp.exists(seg_model_dir):
            raise Exception("Model path {} does not exist".format(
                seg_model_dir))
        self.detector = pdx.load_model(det_model_dir)
        self.segmenter = pdx.load_model(seg_model_dir)

    def decode(self, img_file):
        """图像解码

        参数：
            img_file (str|np.array): 图像路径，或者是已解码的BGR图像数组。

        返回：
            img (np.array): BGR图像数组。
        """

        if isinstance(img_file, str):
            img = cv2.imread(img_file).astype('float32')
        else:
            img = img_file.copy()
        return img

    def filter_bboxes(self, det_results, score_threshold):
        """过滤置信度低于阈值的检测框

        参数：
            det_results (list[dict]): 检测模型预测接口的返回值。
            score_threshold (float)：置信度阈值。

        返回：
            filtered_results (list[dict]): 过滤后的检测狂。

        """
        filtered_results = list()
        for res in det_results:
            if res['score'] > score_threshold:
                filtered_results.append(res)
        return filtered_results

    def roi_crop(self, img, det_results):
        """抠取图像上各检测框的图像区域

        参数：
            img (np.array)：BRG图像数组。
            det_results (list[dict]): 检测模型预测接口的返回值。

        返回：
            sub_imgs (list[np.array]): 各检测框的图像区域。

        """
        sub_imgs = []
        for res in det_results:
            # Crop the bbox area
            xmin, ymin, w, h = res['bbox']
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(img.shape[1], int(xmin + w - 1))
            ymax = min(img.shape[0], int(ymin + h - 1))
            sub_img = img[ymin:(ymax + 1), xmin:(xmax + 1), :]
            sub_imgs.append(sub_img)
        return sub_imgs

    def resize(self, imgs, target_size, interp=cv2.INTER_LINEAR):
        """图像缩放至固定大小

        参数：
            imgs (list[np.array])：批量BGR图像数组。
            target_size (list|tuple)：缩放后的图像大小，格式为[高, 宽]。
            interp (int)：图像差值方法。默认值为cv2.INTER_LINEAR。

        返回：
            resized_imgs (list[np.array])：缩放后的批量BGR图像数组。

        """

        resized_imgs = list()
        for img in imgs:
            img_shape = img.shape
            scale_x = float(target_size[1]) / float(img_shape[1])
            scale_y = float(target_size[0]) / float(img_shape[0])
            resize_img = cv2.resize(
                img, None, None, fx=scale_x, fy=scale_y, interpolation=interp)
            resized_imgs.append(resize_img)
        return resized_imgs

    def seg_predict(self, segmenter, imgs, batch_size):
        """分割模型完成预测

        参数：
            segmenter (pdx.seg.model)：加载后的分割模型。
            imgs (list[np.array])：待预测的输入BGR图像数组。
            batch_size (int): 分割模型前向预测一次时输入图像的批量大小。

        返回：
            seg_results (list[dict]): 输入图像的预测结果。

        """
        seg_results = list()
        num_imgs = len(imgs)
        for i in range(0, num_imgs, batch_size):
            batch = imgs[i:min(num_imgs, i + batch_size)]
            result = segmenter.predict(batch)
            seg_results.extend(result)
        return seg_results

    def erode(self, seg_results, erode_kernel):
        """对分割模型预测结果中label_map做图像腐蚀操作

        参数：
            seg_results (list[dict])：分割模型的预测结果。
            erode_kernel (int): 图像腐蚀的卷积核的大小。

        返回：
            eroded_results (list[dict])：对label_map进行腐蚀后的分割模型预测结果。

        """
        kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
        eroded_results = seg_results
        for i in range(len(seg_results)):
            eroded_results[i]['label_map'] = cv2.erode(
                seg_results[i]['label_map'], kernel)
        return eroded_results

    def circle_to_rectangle(self, seg_results):
        """将圆形表盘的预测结果label_map转换成矩形

        圆形到矩形的计算方法：
            因本案例中两种表盘的刻度起始值都在左下方，故以圆形的中心点为坐标原点，
            从-y轴开始逆时针计算极坐标到x-y坐标的对应关系：
              x = r + r * cos(theta)
              y = r - r * sin(theta)
            注意：
                1. 因为是从-y轴开始逆时针计算，所以r * sin(theta)前有负号。
                2. 还是因为从-y轴开始逆时针计算，所以矩形从上往下对应圆形从外到内，
                   可以想象把圆形从-y轴切开再往左右拉平时，圆形的外围是上面，內围在下面。

        参数：
            seg_results (list[dict])：分割模型的预测结果。

        返回值：
            rectangle_meters (list[np.array])：矩形表盘的预测结果label_map。

        """
        rectangle_meters = list()
        for i, seg_result in enumerate(seg_results):
            label_map = seg_result['label_map']
            # rectangle_meter的大小已经由预先设置的全局变量RECTANGLE_HEIGHT, RECTANGLE_WIDTH决定
            rectangle_meter = np.zeros(
                (RECTANGLE_HEIGHT, RECTANGLE_WIDTH), dtype=np.uint8)
            for row in range(RECTANGLE_HEIGHT):
                for col in range(RECTANGLE_WIDTH):
                    theta = PI * 2 * (col + 1) / RECTANGLE_WIDTH
                    # 矩形从上往下对应圆形从外到内
                    rho = CIRCLE_RADIUS - row - 1
                    y = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.5)
                    x = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.5)
                    rectangle_meter[row, col] = label_map[y, x]
            rectangle_meters.append(rectangle_meter)
        return rectangle_meters

    def rectangle_to_line(self, rectangle_meters):
        """从矩形表盘的预测结果中提取指针和刻度预测结果并沿高度方向压缩成线状格式。

        参数：
            rectangle_meters (list[np.array])：矩形表盘的预测结果label_map。

        返回：
            line_scales (list[np.array])：刻度的线状预测结果。
            line_pointers (list[np.array])：指针的线状预测结果。

        """
        line_scales = list()
        line_pointers = list()

        for rectangle_meter in rectangle_meters:
            height, width = rectangle_meter.shape[0:2]
            line_scale = np.zeros((width), dtype=np.uint8)
            line_pointer = np.zeros((width), dtype=np.uint8)
            for col in range(width):
                for row in range(height):
                    if rectangle_meter[row, col] == SEG_CNAME2CLSID['pointer']:
                        line_pointer[col] += 1
                    elif rectangle_meter[row, col] == SEG_CNAME2CLSID['scale']:
                        line_scale[col] += 1
            line_scales.append(line_scale)
            line_pointers.append(line_pointer)
        return line_scales, line_pointers

    def mean_binarization(self, data_list):
        """对图像进行均值二值化操作

        参数：
            data_list (list[np.array])：待二值化的批量数组。

        返回：
            binaried_data_list (list[np.array])：二值化后的批量数组。

        """
        batch_size = len(data_list)
        binaried_data_list = data_list
        for i in range(batch_size):
            mean_data = np.mean(data_list[i])
            width = data_list[i].shape[0]
            for col in range(width):
                if data_list[i][col] < mean_data:
                    binaried_data_list[i][col] = 0
                else:
                    binaried_data_list[i][col] = 1
        return binaried_data_list

    def locate_scale(self, line_scales):
        """在线状预测结果中找到每根刻度的中心位置

        参数：
            line_scales (list[np.array])：批量的二值化后的刻度线状预测结果。

        返回：
            scale_locations (list[list])：各图像中每根刻度的中心位置。

        """
        batch_size = len(line_scales)
        scale_locations = list()
        for i in range(batch_size):
            line_scale = line_scales[i]
            width = line_scale.shape[0]
            find_start = False
            one_scale_start = 0
            one_scale_end = 0
            locations = list()
            for j in range(width - 1):
                if line_scale[j] > 0 and line_scale[j + 1] > 0:
                    if find_start == False:
                        one_scale_start = j
                        find_start = True
                if find_start:
                    if line_scale[j] == 0 and line_scale[j + 1] == 0:
                        one_scale_end = j - 1
                        one_scale_location = (
                            one_scale_start + one_scale_end) / 2
                        locations.append(one_scale_location)
                        one_scale_start = 0
                        one_scale_end = 0
                        find_start = False
            scale_locations.append(locations)
        return scale_locations

    def locate_pointer(self, line_pointers):
        """在线状预测结果中找到指针的中心位置

        参数：
            line_scales (list[np.array])：批量的指针线状预测结果。

        返回：
            scale_locations (list[list])：各图像中指针的中心位置。

        """
        batch_size = len(line_pointers)
        pointer_locations = list()
        for i in range(batch_size):
            line_pointer = line_pointers[i]
            find_start = False
            pointer_start = 0
            pointer_end = 0
            location = 0
            width = line_pointer.shape[0]
            for j in range(width - 1):
                if line_pointer[j] > 0 and line_pointer[j + 1] > 0:
                    if find_start == False:
                        pointer_start = j
                        find_start = True
                if find_start:
                    if line_pointer[j] == 0 and line_pointer[j + 1] == 0:
                        pointer_end = j - 1
                        location = (pointer_start + pointer_end) / 2
                        find_start = False
                        break
            pointer_locations.append(location)
        return pointer_locations

    def get_relative_location(self, scale_locations, pointer_locations):
        """找到指针指向了第几根刻度

        参数：
            scale_locations (list[list])：批量的每根刻度的中心点位置。
            pointer_locations (list[list])：批量的指针的中心点位置。

        返回：
            pointed_scales (list[dict])：每个表的结果组成的list。每个表的结果由字典表示，
                字典有两个关键词：'num_scales'、'pointed_scale'，分别表示预测的刻度根数、
                预测的指针指向了第几根刻度。

        """

        pointed_scales = list()
        for scale_location, pointer_location in zip(scale_locations,
                                                    pointer_locations):
            num_scales = len(scale_location)
            pointed_scale = -1
            if num_scales > 0:
                for i in range(num_scales - 1):
                    if scale_location[
                            i] <= pointer_location and pointer_location < scale_location[
                                i + 1]:
                        pointed_scale = i + (
                            pointer_location - scale_location[i]
                        ) / (scale_location[i + 1] - scale_location[i] + 1e-05
                             ) + 1
            result = {'num_scales': num_scales, 'pointed_scale': pointed_scale}
            pointed_scales.append(result)
        return pointed_scales

    def calculate_reading(self, pointed_scales):
        """根据刻度的间隔值和指针指向的刻度根数计算表盘的读数
        """
        readings = list()
        batch_size = len(pointed_scales)
        for i in range(batch_size):
            pointed_scale = pointed_scales[i]
            # 刻度根数大于阈值的为第一种表盘
            if pointed_scale['num_scales'] > TYPE_THRESHOLD:
                reading = pointed_scale['pointed_scale'] * METER_CONFIG[0][
                    'scale_interval_value']
            else:
                reading = pointed_scale['pointed_scale'] * METER_CONFIG[1][
                    'scale_interval_value']
            readings.append(reading)

        return readings

    def get_meter_reading(self, seg_results):
        """对分割结果进行读数后处理得到各表盘的读数

        参数：
            seg_results (list[dict]): 分割模型的预测结果。

        返回：
            meter_readings (list[dcit]): 各表盘的读数。

        """

        rectangle_meters = self.circle_to_rectangle(seg_results)
        line_scales, line_pointers = self.rectangle_to_line(rectangle_meters)
        binaried_scales = self.mean_binarization(line_scales)
        binaried_pointers = self.mean_binarization(line_pointers)
        scale_locations = self.locate_scale(binaried_scales)
        pointer_locations = self.locate_pointer(binaried_pointers)
        pointed_scales = self.get_relative_location(scale_locations,
                                                    pointer_locations)
        meter_readings = self.calculate_reading(pointed_scales)
        return meter_readings

    def print_meter_readings(self, meter_readings):
        """打印各表盘的读数

        参数：
            meter_readings (list[dict])：各表盘的读数
        """
        for i in range(len(meter_readings)):
            print("Meter {}: {}".format(i + 1, meter_readings[i]))

    def visualize(self, img, det_results, meter_readings, save_dir="./"):
        """可视化图像中各表盘的位置和读数

        参数：
            img (str|np.array): 图像路径，或者是已解码的BGR图像数组。
            det_results (dict): 检测模型的预测结果。
            meter_readings (list): 各表盘的读数。
            save_dir (str)：可视化后的图片保存路径。

        """
        vis_results = list()
        for i, res in enumerate(det_results):
            # 将检测结果中的关键词`score`替换成读数，就可以调用pdx.det.visualize画图了
            res['score'] = meter_readings[i]
            vis_results.append(res)
        # 检测结果可视化时会滤除score低于threshold的框，这里读数都是>=-1的，所以设置thresh=-1
        pdx.det.visualize(img, vis_results, threshold=-1, save_dir=save_dir)

    def predict(self,
                img_file,
                save_dir='./',
                use_erode=True,
                erode_kernel=4,
                score_threshold=0.5,
                seg_batch_size=2):
        """检测图像中的表盘，而后分割出各表盘中的指针和刻度，对分割结果进行读数后处理后得到各表盘的读数。


        参数：
            img_file (str)：待预测的图片路径。
            save_dir (str): 可视化结果的保存路径。
            use_erode (bool, optional): 是否对分割预测结果做图像腐蚀。默认值：True。
            erode_kernel (int, optional): 图像腐蚀的卷积核大小。默认值: 4。
            score_threshold (float, optional): 用于滤除检测框的置信度阈值。默认值：0.5。
            seg_batch_size (int, optional)：分割模型前向推理一次时输入表盘图像的批量大小。默认值为：2。
        """

        img = self.decode(img_file)
        det_results = self.detector.predict(img)
        filtered_results = self.filter_bboxes(det_results, score_threshold)
        sub_imgs = self.roi_crop(img, filtered_results)
        sub_imgs = self.resize(sub_imgs, METER_SHAPE)
        seg_results = self.seg_predict(self.segmenter, sub_imgs,
                                       seg_batch_size)
        seg_results = self.erode(seg_results, erode_kernel)
        meter_readings = self.get_meter_reading(seg_results)
        self.print_meter_readings(meter_readings)
        self.visualize(img, filtered_results, meter_readings, save_dir)


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

    meter_reader = MeterReader(args.det_model_dir, args.seg_model_dir)
    if len(image_lists) > 0:
        for image in image_lists:
            meter_reader.predict(image, args.save_dir, args.use_erode,
                                 args.erode_kernel, args.score_threshold,
                                 args.seg_batch_size)


if __name__ == '__main__':
    args = parse_args()
    infer(args)
