# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import copy


def execute_imgaug(augmenter, im, bboxes=None, polygons=None,
                   segment_map=None):
    # 预处理，将bboxes, polygons转换成imgaug格式
    import imgaug.augmentables.kps as kps
    import imgaug.augmentables.bbs as bbs

    aug_im = im.astype('uint8')
    aug_im = augmenter.augment(image=aug_im).astype('float32')
    return aug_im

    # TODO imgaug的标注处理逻辑与paddlex已存的transform存在部分差异
    # 目前仅支持对原图进行处理，因此只能使用pixlevel的imgaug增强操作
    # 以下代码暂不会执行
    aug_bboxes = None
    if bboxes is not None:
        aug_bboxes = list()
        for i in range(len(bboxes)):
            x1 = bboxes[i, 0]
            y1 = bboxes[i, 1]
            x2 = bboxes[i, 2]
            y2 = bboxes[i, 3]
            aug_bboxes.append(bbs.BoundingBox(x1, y1, x2, y2))

    aug_points = None
    if polygons is not None:
        aug_points = list()
        for i in range(len(polygons)):
            num = len(polygons[i])
            for j in range(num):
                tmp = np.reshape(polygons[i][j], (-1, 2))
                for k in range(len(tmp)):
                    aug_points.append(kps.Keypoint(tmp[k, 0], tmp[k, 1]))

    aug_segment_map = None
    if segment_map is not None:
        if len(segment_map.shape) == 2:
            h, w = segment_map.shape
            aug_segment_map = np.reshape(segment_map, (1, h, w, 1))
        elif len(segment_map.shape) == 3:
            h, w, c = segment_map.shape
            aug_segment_map = np.reshape(segment_map, (1, h, w, c))
        else:
            raise Exception(
                "Only support 2-dimensions for 3-dimensions for segment_map")

    unnormalized_batch = augmenter.augment(
        image=aug_im,
        bounding_boxes=aug_bboxes,
        keypoints=aug_points,
        segmentation_maps=aug_segment_map,
        return_batch=True)
    aug_im = unnormalized_batch.images_aug[0]
    aug_bboxes = unnormalized_batch.bounding_boxes_aug
    aug_points = unnormalized_batch.keypoints_aug
    aug_seg_map = unnormalized_batch.segmentation_maps_aug

    aug_im = aug_im.astype('float32')

    if aug_bboxes is not None:
        converted_bboxes = list()
        for i in range(len(aug_bboxes)):
            converted_bboxes.append([
                aug_bboxes[i].x1, aug_bboxes[i].y1, aug_bboxes[i].x2,
                aug_bboxes[i].y2
            ])
        aug_bboxes = converted_bboxes

    aug_polygons = None
    if aug_points is not None:
        aug_polygons = copy.deepcopy(polygons)
        idx = 0
        for i in range(len(aug_polygons)):
            num = len(aug_polygons[i])
            for j in range(num):
                num_points = len(aug_polygons[i][j]) // 2
                for k in range(num_points):
                    aug_polygons[i][j][k * 2] = aug_points[idx].x
                    aug_polygons[i][j][k * 2 + 1] = aug_points[idx].y
                    idx += 1

    result = [aug_im]
    if aug_bboxes is not None:
        result.append(np.array(aug_bboxes))
    if aug_polygons is not None:
        result.append(aug_polygons)
    if aug_seg_map is not None:
        n, h, w, c = aug_seg_map.shape
        if len(segment_map.shape) == 2:
            aug_seg_map = np.reshape(aug_seg_map, (h, w))
        elif len(segment_map.shape) == 3:
            aug_seg_map = np.reshape(aug_seg_map, (h, w, c))
        result.append(aug_seg_map)
    return result
