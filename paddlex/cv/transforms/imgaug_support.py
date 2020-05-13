# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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


def execute_imgaug(augmenter, im, bboxes=None, polygons=None,
                   segment_map=None):
    # 预处理，将bboxes, polygons转换成imgaug格式
    import imgaug.augmentables.polys as polys
    import imgaug.augmentables.bbs as bbs

    aug_im = im.astype('uint8')

    aug_bboxes = None
    if bboxes is not None:
        aug_bboxes = list()
        for i in range(len(bboxes)):
            x1 = bboxes[i, 0] - 1
            y1 = bboxes[i, 1]
            x2 = bboxes[i, 2]
            y2 = bboxes[i, 3]
            aug_bboxes.append(bbs.BoundingBox(x1, y1, x2, y2))

    aug_polygons = None
    lod_info = list()
    if polygons is not None:
        aug_polygons = list()
        for i in range(len(polygons)):
            num = len(polygons[i])
            lod_info.append(num)
            for j in range(num):
                points = np.reshape(polygons[i][j], (-1, 2))
                aug_polygons.append(polys.Polygon(points))

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

    aug_im, aug_bboxes, aug_polygons, aug_seg_map = augmenter.augment(
        image=aug_im,
        bounding_boxes=aug_bboxes,
        polygons=aug_polygons,
        segmentation_maps=aug_segment_map)

    aug_im = aug_im.astype('float32')

    if aug_polygons is not None:
        assert len(aug_bboxes) == len(
            lod_info
        ), "Number of aug_bboxes should be equal to number of aug_polygons"

    if aug_bboxes is not None:
        # 裁剪掉在图像之外的bbox和polygon
        for i in range(len(aug_bboxes)):
            aug_bboxes[i] = aug_bboxes[i].clip_out_of_image(aug_im)
        if aug_polygons is not None:
            for i in range(len(aug_polygons)):
                aug_polygons[i] = aug_polygons[i].clip_out_of_image(aug_im)

        # 过滤掉无效的bbox和polygon，并转换为训练数据格式
        converted_bboxes = list()
        converted_polygons = list()
        poly_index = 0
        for i in range(len(aug_bboxes)):
            # 过滤width或height不足1像素的框
            if aug_bboxes[i].width < 1 or aug_bboxes[i].height < 1:
                continue
            if aug_polygons is None:
                converted_bboxes.append([
                    aug_bboxes[i].x1, aug_bboxes[i].y1, aug_bboxes[i].x2,
                    aug_bboxes[i].y2
                ])
                continue

            # 如若有polygons，将会继续执行下面代码
            polygons_this_box = list()
            for ps in aug_polygons[poly_index:poly_index + lod_info[i]]:
                if len(ps) == 0:
                    continue
                for p in ps:
                    # 没有3个point的polygon被过滤
                    if len(p.exterior) < 3:
                        continue
                    polygons_this_box.append(p.exterior.flatten().tolist())
            poly_index += lod_info[i]

            if len(polygons_this_box) == 0:
                continue
            converted_bboxes.append([
                aug_bboxes[i].x1, aug_bboxes[i].y1, aug_bboxes[i].x2,
                aug_bboxes[i].y2
            ])
            converted_polygons.append(polygons_this_box)
        if len(converted_bboxes) == 0:
            aug_im = im
            converted_bboxes = bboxes
            converted_polygons = polygons

    result = [aug_im]
    if bboxes is not None:
        result.append(np.array(converted_bboxes))
    if polygons is not None:
        result.append(converted_polygons)
    if segment_map is not None:
        n, h, w, c = aug_seg_map.shape
        if len(segment_map.shape) == 2:
            aug_seg_map = np.reshape(aug_seg_map, (h, w))
        elif len(segment_map.shape) == 3:
            aug_seg_map = np.reshape(aug_seg_map, (h, w, c))
        result.append(aug_seg_map)
    return result
