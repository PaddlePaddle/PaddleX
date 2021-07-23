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

import copy
import os
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import kmeans
from paddlex.utils import logging


class BaseAnchorCluster(object):
    def __init__(self, num_anchors, cache, cache_path, verbose=True):
        """
        Base Anchor Cluster
        Args:
            num_anchors (int): number of clusters
            cache (bool): whether using cache
            cache_path (str): cache directory path
            verbose (bool): whether print results
        """
        super(BaseAnchorCluster, self).__init__()
        self.num_anchors = num_anchors
        self.cache_path = cache_path
        self.cache = cache
        self.verbose = verbose

    def print_result(self, centers):
        raise NotImplementedError('%s.print_result is not available' %
                                  self.__class__.__name__)

    def get_whs(self):
        whs_cache_path = os.path.join(self.cache_path, 'whs.npy')
        shapes_cache_path = os.path.join(self.cache_path, 'shapes.npy')
        if self.cache and os.path.exists(whs_cache_path) and os.path.exists(
                shapes_cache_path):
            self.whs = np.load(whs_cache_path)
            self.shapes = np.load(shapes_cache_path)
            return self.whs, self.shapes
        whs = np.zeros((0, 2))
        shapes = np.zeros((0, 2))
        samples = copy.deepcopy(self.dataset.file_list)
        for sample in tqdm(samples):
            im_h, im_w = sample['image_shape']
            bbox = sample['gt_bbox']
            wh = bbox[:, 2:4] - bbox[:, 0:2]
            wh = wh / np.array([[im_w, im_h]])
            shape = np.ones_like(wh) * np.array([[im_w, im_h]])
            whs = np.vstack((whs, wh))
            shapes = np.vstack((shapes, shape))

        if self.cache:
            os.makedirs(self.cache_path, exist_ok=True)
            np.save(whs_cache_path, whs)
            np.save(shapes_cache_path, shapes)

        self.whs = whs
        self.shapes = shapes
        return self.whs, self.shapes

    def calc_anchors(self):
        raise NotImplementedError('%s.calc_anchors is not available' %
                                  self.__class__.__name__)

    def __call__(self):
        self.get_whs()
        centers = self.calc_anchors()
        if self.verbose:
            self.print_result(centers)
        return centers


class YOLOAnchorCluster(BaseAnchorCluster):
    def __init__(self,
                 num_anchors,
                 dataset,
                 image_size,
                 cache,
                 cache_path=None,
                 iters=300,
                 gen_iters=1000,
                 thresh=0.25,
                 verbose=True):
        """
        YOLOv5 Anchor Cluster

        Reference:
            https://github.com/ultralytics/yolov5/blob/master/utils/autoanchor.py

        Args:
            num_anchors (int): number of clusters
            dataset (DataSet): DataSet instance, VOC or COCO
            image_size (list or int): [h, w], being an int means image height and image width are the same.
            cache (bool): whether using cache
            cache_path (str or None, optional): cache directory path. If None, use `data_dir` of dataset.
            iters (int, optional): iters of kmeans algorithm
            gen_iters (int, optional): iters of genetic algorithm
            threshold (float, optional): anchor scale threshold
            verbose (bool, optional): whether print results
        """
        self.dataset = dataset
        if cache_path is None:
            cache_path = self.dataset.data_dir
        if isinstance(image_size, int):
            image_size = [image_size] * 2
        self.image_size = image_size
        self.iters = iters
        self.gen_iters = gen_iters
        self.thresh = thresh
        super(YOLOAnchorCluster, self).__init__(
            num_anchors, cache, cache_path, verbose=verbose)

    def print_result(self, centers):
        whs = self.whs
        x, best = self.metric(whs, centers)
        bpr, aat = (best > self.thresh).mean(), (
            x > self.thresh).mean() * self.num_anchors
        logging.info(
            'thresh=%.2f: %.4f best possible recall, %.2f anchors past thr' %
            (self.thresh, bpr, aat))
        logging.info(
            'n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thresh=%.3f-mean: '
            % (self.num_anchors, self.image_size, x.mean(), best.mean(),
               x[x > self.thresh].mean()))
        logging.info('%d anchor cluster result: [w, h]' % self.num_anchors)
        for w, h in centers:
            logging.info('[%d, %d]' % (w, h))

    def metric(self, whs, centers):
        r = whs[:, None] / centers[None]
        x = np.minimum(r, 1. / r).min(2)
        return x, x.max(1)

    def fitness(self, whs, centers):
        _, best = self.metric(whs, centers)
        return (best * (best > self.thresh)).mean()

    def calc_anchors(self):
        self.whs = self.whs * self.shapes / self.shapes.max(
            1, keepdims=True) * np.array([self.image_size[::-1]])
        wh0 = self.whs
        i = (wh0 < 3.0).any(1).sum()
        if i:
            logging.warning('Extremely small objects found. %d of %d '
                            'labels are < 3 pixels in width or height' %
                            (i, len(wh0)))

        wh = wh0[(wh0 >= 2.0).any(1)]
        logging.info('Running kmeans for %g anchors on %g points...' %
                     (self.num_anchors, len(wh)))
        s = wh.std(0)
        centers, dist = kmeans(wh / s, self.num_anchors, iter=self.iters)
        centers *= s

        f, sh, mp, s = self.fitness(wh, centers), centers.shape, 0.9, 0.1
        pbar = tqdm(
            range(self.gen_iters),
            desc='Evolving anchors with Genetic Algorithm')
        for _ in pbar:
            v = np.ones(sh)
            while (v == 1).all():
                v = ((np.random.random(sh) < mp) * np.random.random() *
                     np.random.randn(*sh) * s + 1).clip(0.3, 3.0)
            new_centers = (centers.copy() * v).clip(min=2.0)
            new_f = self.fitness(wh, new_centers)
            if new_f > f:
                f, centers = new_f, new_centers.copy()
                pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f

        centers = np.round(centers[np.argsort(centers.prod(1))]).astype(int)
        return centers
