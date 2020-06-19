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

import argparse
import os
import os.path as osp
import cv2
import numpy as np

from postprocess import postprocess, threshold_mask
import paddlex as pdx
import paddlex.utils.logging as logging
from paddlex.seg import transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description='HumanSeg inference for video')
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        help='Model path for inference',
        type=str)
    parser.add_argument(
        '--video_path',
        dest='video_path',
        help='Video path for inference, camera will be used if the path not existing',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='./output')
    parser.add_argument(
        "--image_shape",
        dest="image_shape",
        help="The image shape for net inputs.",
        nargs=2,
        default=[192, 192],
        type=int)

    return parser.parse_args()


def recover(img, im_info):
    if im_info[0] == 'resize':
        w, h = im_info[1][1], im_info[1][0]
        img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
    elif im_info[0] == 'padding':
        w, h = im_info[1][0], im_info[1][0]
        img = img[0:h, 0:w, :]
    return img


def video_infer(args):
    resize_h = args.image_shape[1]
    resize_w = args.image_shape[0]

    test_transforms = transforms.Compose([transforms.Normalize()])
    model = pdx.load_model(args.model_dir)
    if not args.video_path:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError("Error opening video stream or file, "
                      "--video_path whether existing: {}"
                      " or camera whether working".format(args.video_path))
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    prev_gray = np.zeros((resize_h, resize_w), np.uint8)
    prev_cfd = np.zeros((resize_h, resize_w), np.float32)
    is_init = True

    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.video_path:
        logging.info("Please wait. It is computing......")
        # 用于保存预测结果视频
        if not osp.exists(args.save_dir):
            os.makedirs(args.save_dir)
        out = cv2.VideoWriter(
            osp.join(args.save_dir, 'result.avi'),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
        # 开始获取视频帧
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                im_shape = frame.shape
                im_scale_x = float(resize_w) / float(im_shape[1])
                im_scale_y = float(resize_h) / float(im_shape[0])
                im = cv2.resize(
                    frame,
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=cv2.INTER_LINEAR)
                image = im.astype('float32')
                im_info = ('resize', im_shape[0:2])
                pred = model.predict(image)
                score_map = pred['score_map']
                cur_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                score_map = 255 * score_map[:, :, 1]
                optflow_map = postprocess(cur_gray, score_map, prev_gray, prev_cfd, \
                        disflow, is_init)
                prev_gray = cur_gray.copy()
                prev_cfd = optflow_map.copy()
                is_init = False
                optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
                optflow_map = threshold_mask(
                    optflow_map, thresh_bg=0.2, thresh_fg=0.8)
                img_matting = np.repeat(
                    optflow_map[:, :, np.newaxis], 3, axis=2)
                img_matting = recover(img_matting, im_info)
                bg_im = np.ones_like(img_matting) * 255
                comb = (img_matting * frame +
                        (1 - img_matting) * bg_im).astype(np.uint8)
                out.write(comb)
            else:
                break
        cap.release()
        out.release()

    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                im_shape = frame.shape
                im_scale_x = float(resize_w) / float(im_shape[1])
                im_scale_y = float(resize_h) / float(im_shape[0])
                im = cv2.resize(
                    frame,
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=cv2.INTER_LINEAR)
                image = im.astype('float32')
                im_info = ('resize', im_shape[0:2])
                pred = model.predict(image)
                score_map = pred['score_map']
                cur_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                cur_gray = cv2.resize(cur_gray, (resize_w, resize_h))
                score_map = 255 * score_map[:, :, 1]
                optflow_map = postprocess(cur_gray, score_map, prev_gray, prev_cfd, \
                                          disflow, is_init)
                prev_gray = cur_gray.copy()
                prev_cfd = optflow_map.copy()
                is_init = False
                optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
                optflow_map = threshold_mask(
                    optflow_map, thresh_bg=0.2, thresh_fg=0.8)
                img_matting = np.repeat(
                    optflow_map[:, :, np.newaxis], 3, axis=2)
                img_matting = recover(img_matting, im_info)
                bg_im = np.ones_like(img_matting) * 255
                comb = (img_matting * frame +
                        (1 - img_matting) * bg_im).astype(np.uint8)
                cv2.imshow('HumanSegmentation', comb)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()


if __name__ == "__main__":
    args = parse_args()
    video_infer(args)
