# coding: utf8
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

from __future__ import absolute_import

import numpy as np
import json
import os
import sys
import cv2
import copy
import paddlex.utils.logging as logging

# fix linspace problem for pycocotools while numpy > 1.17.2
backup_linspace = np.linspace


def fixed_linspace(start,
                   stop,
                   num=50,
                   endpoint=True,
                   retstep=False,
                   dtype=None,
                   axis=0):
    num = int(num)
    return backup_linspace(start, stop, num, endpoint, retstep, dtype, axis)


def eval_results(results,
                 metric,
                 coco_gt,
                 with_background=True,
                 resolution=None,
                 is_bbox_normalized=False,
                 map_type='11point'):
    """Evaluation for evaluation program results"""
    box_ap_stats = []
    coco_gt_data = copy.deepcopy(coco_gt)
    eval_details = {'gt': copy.deepcopy(coco_gt.dataset)}
    if metric == 'COCO':
        np.linspace = fixed_linspace
        if 'proposal' in results[0]:
            proposal_eval(results, coco_gt_data)
        if 'bbox' in results[0]:
            box_ap_stats, xywh_results = coco_bbox_eval(
                results,
                coco_gt_data,
                with_background,
                is_bbox_normalized=is_bbox_normalized)

        if 'mask' in results[0]:
            mask_ap_stats, segm_results = mask_eval(results, coco_gt_data,
                                                    resolution)
            ap_stats = [box_ap_stats, mask_ap_stats]
            eval_details['bbox'] = xywh_results
            eval_details['mask'] = segm_results
            return ap_stats, eval_details
        np.linspace = backup_linspace
    else:
        if 'accum_map' in results[-1]:
            res = np.mean(results[-1]['accum_map'][0])
            logging.debug('mAP: {:.2f}'.format(res * 100.))
            box_ap_stats.append(res * 100.)
        elif 'bbox' in results[0]:
            box_ap, xywh_results = voc_bbox_eval(
                results,
                coco_gt_data,
                with_background,
                is_bbox_normalized=is_bbox_normalized,
                map_type=map_type)
            box_ap_stats.append(box_ap)
    eval_details['bbox'] = xywh_results
    return box_ap_stats, eval_details


def proposal_eval(results, coco_gt, outputfile, max_dets=(100, 300, 1000)):
    assert 'proposal' in results[0]
    assert outfile.endswith('.json')

    xywh_results = proposal2out(results)
    assert len(
        xywh_results) > 0, "The number of valid proposal detected is zero.\n \
        Please use reasonable model and check input data."

    with open(outfile, 'w') as f:
        json.dump(xywh_results, f)

    cocoapi_eval(xywh_results, 'proposal', coco_gt=coco_gt, max_dets=max_dets)
    # flush coco evaluation result
    sys.stdout.flush()


def coco_bbox_eval(results,
                   coco_gt,
                   with_background=True,
                   is_bbox_normalized=False):
    assert 'bbox' in results[0]
    # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    # or matplotlib.backends is imported for the first time
    # pycocotools import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    from pycocotools.coco import COCO

    cat_ids = coco_gt.getCatIds()

    # when with_background = True, mapping category to classid, like:
    #   background:0, first_class:1, second_class:2, ...
    clsid2catid = dict(
        {i + int(with_background): catid
         for i, catid in enumerate(cat_ids)})

    xywh_results = bbox2out(
        results, clsid2catid, is_bbox_normalized=is_bbox_normalized)

    results = copy.deepcopy(xywh_results)
    if len(xywh_results) == 0:
        logging.warning(
            "The number of valid bbox detected is zero.\n Please use reasonable model and check input data.\n stop eval!"
        )
        return [0.0], results

    map_stats = cocoapi_eval(xywh_results, 'bbox', coco_gt=coco_gt)
    # flush coco evaluation result
    sys.stdout.flush()
    return map_stats, results


def loadRes(coco_obj, anns):
    """
    Load result file and return a result api object.
    :param   resFile (str)     : file name of result file
    :return: res (obj)         : result api object
    """
    # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    # or matplotlib.backends is imported for the first time
    # pycocotools import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    from pycocotools.coco import COCO
    import pycocotools.mask as maskUtils
    import time
    res = COCO()
    res.dataset['images'] = [img for img in coco_obj.dataset['images']]

    tic = time.time()
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(coco_obj.getImgIds())), \
           'Results do not correspond to current coco set'
    if 'caption' in anns[0]:
        imgIds = set([img['id'] for img in res.dataset['images']]) & set(
            [ann['image_id'] for ann in anns])
        res.dataset['images'] = [
            img for img in res.dataset['images'] if img['id'] in imgIds
        ]
        for id, ann in enumerate(anns):
            ann['id'] = id + 1
    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(coco_obj.dataset[
            'categories'])
        for id, ann in enumerate(anns):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if not 'segmentation' in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'segmentation' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(coco_obj.dataset[
            'categories'])
        for id, ann in enumerate(anns):
            # now only support compressed RLE format as segmentation results
            ann['area'] = maskUtils.area(ann['segmentation'])
            if not 'bbox' in ann:
                ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    elif 'keypoints' in anns[0]:
        res.dataset['categories'] = copy.deepcopy(coco_obj.dataset[
            'categories'])
        for id, ann in enumerate(anns):
            s = ann['keypoints']
            x = s[0::3]
            y = s[1::3]
            x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann['area'] = (x1 - x0) * (y1 - y0)
            ann['id'] = id + 1
            ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]

    res.dataset['annotations'] = anns
    res.createIndex()
    return res


def mask_eval(results, coco_gt, resolution, thresh_binarize=0.5):
    assert 'mask' in results[0]
    # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    # or matplotlib.backends is imported for the first time
    # pycocotools import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    from pycocotools.coco import COCO

    clsid2catid = {i + 1: v for i, v in enumerate(coco_gt.getCatIds())}

    segm_results = mask2out(results, clsid2catid, resolution, thresh_binarize)
    results = copy.deepcopy(segm_results)
    if len(segm_results) == 0:
        logging.warning(
            "The number of valid mask detected is zero.\n Please use reasonable model and check input data."
        )
        return None, results

    map_stats = cocoapi_eval(segm_results, 'segm', coco_gt=coco_gt)
    return map_stats, results


def cocoapi_eval(anns,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000)):
    """
    Args:
        anns: Evaluation result.
        style: COCOeval style, can be `bbox` , `segm` and `proposal`.
        coco_gt: Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file: COCO annotations file.
        max_dets: COCO evaluation maxDets.
    """
    assert coco_gt != None or anno_file != None
    # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    # or matplotlib.backends is imported for the first time
    # pycocotools import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if coco_gt == None:
        coco_gt = COCO(anno_file)
    logging.debug("Start evaluate...")
    coco_dt = loadRes(coco_gt, anns)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def proposal2out(results, is_bbox_normalized=False):
    xywh_res = []
    for t in results:
        bboxes = t['proposal'][0]
        lengths = t['proposal'][1][0]
        im_ids = np.array(t['im_id'][0]).flatten()
        assert len(lengths) == im_ids.size
        if bboxes.shape == (1, 1) or bboxes is None:
            continue

        k = 0
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i])
            for j in range(num):
                dt = bboxes[k]
                xmin, ymin, xmax, ymax = dt.tolist()

                if is_bbox_normalized:
                    xmin, ymin, xmax, ymax = \
                            clip_bbox([xmin, ymin, xmax, ymax])
                    w = xmax - xmin
                    h = ymax - ymin
                else:
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1

                bbox = [xmin, ymin, w, h]
                coco_res = {
                    'image_id': im_id,
                    'category_id': 1,
                    'bbox': bbox,
                    'score': 1.0
                }
                xywh_res.append(coco_res)
                k += 1
    return xywh_res


def bbox2out(results, clsid2catid, is_bbox_normalized=False):
    """
    Args:
        results: request a dict, should include: `bbox`, `im_id`,
                 if is_bbox_normalized=True, also need `im_shape`.
        clsid2catid: class id to category id map of COCO2017 dataset.
        is_bbox_normalized: whether or not bbox is normalized.
    """
    xywh_res = []
    for t in results:
        bboxes = t['bbox'][0]
        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0]).flatten()
        if bboxes.shape == (1, 1) or bboxes is None:
            continue

        k = 0
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i])
            for j in range(num):
                dt = bboxes[k]
                clsid, score, xmin, ymin, xmax, ymax = dt.tolist()
                catid = (clsid2catid[int(clsid)])

                if is_bbox_normalized:
                    xmin, ymin, xmax, ymax = \
                            clip_bbox([xmin, ymin, xmax, ymax])
                    w = xmax - xmin
                    h = ymax - ymin
                    im_shape = t['im_shape'][0][i].tolist()
                    im_height, im_width = int(im_shape[0]), int(im_shape[1])
                    xmin *= im_width
                    ymin *= im_height
                    w *= im_width
                    h *= im_height
                else:
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1

                bbox = [xmin, ymin, w, h]
                coco_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'bbox': bbox,
                    'score': score
                }
                xywh_res.append(coco_res)
                k += 1
    return xywh_res


def mask2out(results, clsid2catid, resolution, thresh_binarize=0.5):
    # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    # or matplotlib.backends is imported for the first time
    # pycocotools import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    import pycocotools.mask as mask_util
    scale = (resolution + 2.0) / resolution

    segm_res = []

    # for each batch
    for t in results:
        bboxes = t['bbox'][0]

        lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0])
        if bboxes.shape == (1, 1) or bboxes is None:
            continue
        if len(bboxes.tolist()) == 0:
            continue

        masks = t['mask'][0]

        s = 0
        # for each sample
        for i in range(len(lengths)):
            num = lengths[i]
            im_id = int(im_ids[i][0])
            im_shape = t['im_shape'][0][i]

            bbox = bboxes[s:s + num][:, 2:]
            clsid_scores = bboxes[s:s + num][:, 0:2]
            mask = masks[s:s + num]
            s += num

            im_h = int(im_shape[0])
            im_w = int(im_shape[1])

            expand_bbox = expand_boxes(bbox, scale)
            expand_bbox = expand_bbox.astype(np.int32)

            padded_mask = np.zeros(
                (resolution + 2, resolution + 2), dtype=np.float32)

            for j in range(num):
                xmin, ymin, xmax, ymax = expand_bbox[j].tolist()
                clsid, score = clsid_scores[j].tolist()
                clsid = int(clsid)
                padded_mask[1:-1, 1:-1] = mask[j, clsid, :, :]

                catid = clsid2catid[clsid]

                w = xmax - xmin + 1
                h = ymax - ymin + 1
                w = np.maximum(w, 1)
                h = np.maximum(h, 1)

                resized_mask = cv2.resize(padded_mask, (w, h))
                resized_mask = np.array(
                    resized_mask > thresh_binarize, dtype=np.uint8)
                im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

                x0 = min(max(xmin, 0), im_w)
                x1 = min(max(xmax + 1, 0), im_w)
                y0 = min(max(ymin, 0), im_h)
                y1 = min(max(ymax + 1, 0), im_h)

                im_mask[y0:y1, x0:x1] = resized_mask[(y0 - ymin):(y1 - ymin), (
                    x0 - xmin):(x1 - xmin)]
                segm = mask_util.encode(
                    np.array(
                        im_mask[:, :, np.newaxis], order='F'))[0]
                catid = clsid2catid[clsid]
                segm['counts'] = segm['counts'].decode('utf8')
                coco_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'segmentation': segm,
                    'score': score
                }
                segm_res.append(coco_res)
    return segm_res


def expand_boxes(boxes, scale):
    """
    Expand an array of boxes by a given scale.
    """
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def voc_bbox_eval(results,
                  coco_gt,
                  with_background=False,
                  overlap_thresh=0.5,
                  map_type='11point',
                  is_bbox_normalized=False,
                  evaluate_difficult=False):
    """
    Bounding box evaluation for VOC dataset

    Args:
        results (list): prediction bounding box results.
        class_num (int): evaluation class number.
        overlap_thresh (float): the postive threshold of
                        bbox overlap
        map_type (string): method for mAP calcualtion,
                        can only be '11point' or 'integral'
        is_bbox_normalized (bool): whether bbox is normalized
                        to range [0, 1].
        evaluate_difficult (bool): whether to evaluate
                        difficult gt bbox.
    """
    assert 'bbox' in results[0]

    logging.debug("Start evaluate...")
    # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    # or matplotlib.backends is imported for the first time
    # pycocotools import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    from pycocotools.coco import COCO

    cat_ids = coco_gt.getCatIds()

    # when with_background = True, mapping category to classid, like:
    #   background:0, first_class:1, second_class:2, ...
    clsid2catid = dict(
        {i + int(with_background): catid
         for i, catid in enumerate(cat_ids)})
    class_num = len(clsid2catid) + int(with_background)
    detection_map = DetectionMAP(
        class_num=class_num,
        overlap_thresh=overlap_thresh,
        map_type=map_type,
        is_bbox_normalized=is_bbox_normalized,
        evaluate_difficult=evaluate_difficult)

    xywh_res = []
    det_nums = 0
    gt_nums = 0
    for t in results:
        bboxes = t['bbox'][0]
        bbox_lengths = t['bbox'][1][0]
        im_ids = np.array(t['im_id'][0]).flatten()
        if bboxes.shape == (1, 1) or bboxes is None:
            continue

        gt_boxes = t['gt_box'][0]
        gt_labels = t['gt_label'][0]
        difficults = t['is_difficult'][0] if not evaluate_difficult \
                            else None

        if len(t['gt_box'][1]) == 0:
            # gt_box, gt_label, difficult read as zero padded Tensor
            bbox_idx = 0
            for i in range(len(gt_boxes)):
                gt_box = gt_boxes[i]
                gt_label = gt_labels[i]
                difficult = None if difficults is None \
                                else difficults[i]
                bbox_num = bbox_lengths[i]
                bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
                gt_box, gt_label, difficult = prune_zero_padding(
                    gt_box, gt_label, difficult)
                detection_map.update(bbox, gt_box, gt_label, difficult)
                bbox_idx += bbox_num
                det_nums += bbox_num
                gt_nums += gt_box.shape[0]

                im_id = int(im_ids[i])
                for b in bbox:
                    clsid, score, xmin, ymin, xmax, ymax = b.tolist()
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1
                    bbox = [xmin, ymin, w, h]
                    coco_res = {
                        'image_id': im_id,
                        'category_id': clsid2catid[clsid],
                        'bbox': bbox,
                        'score': score
                    }
                    xywh_res.append(coco_res)
        else:
            # gt_box, gt_label, difficult read as LoDTensor
            gt_box_lengths = t['gt_box'][1][0]
            bbox_idx = 0
            gt_box_idx = 0
            for i in range(len(bbox_lengths)):
                bbox_num = bbox_lengths[i]
                gt_box_num = gt_box_lengths[i]
                bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
                gt_box = gt_boxes[gt_box_idx:gt_box_idx + gt_box_num]
                gt_label = gt_labels[gt_box_idx:gt_box_idx + gt_box_num]
                difficult = None if difficults is None else \
                            difficults[gt_box_idx: gt_box_idx + gt_box_num]
                detection_map.update(bbox, gt_box, gt_label, difficult)
                bbox_idx += bbox_num
                gt_box_idx += gt_box_num

                im_id = int(im_ids[i])
                for b in bbox:
                    clsid, score, xmin, ymin, xmax, ymax = b.tolist()
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1
                    bbox = [xmin, ymin, w, h]
                    coco_res = {
                        'image_id': im_id,
                        'category_id': clsid2catid[clsid],
                        'bbox': bbox,
                        'score': score
                    }
                    xywh_res.append(coco_res)

    logging.debug("Accumulating evaluatation results...")
    detection_map.accumulate()
    map_stat = 100. * detection_map.get_map()
    logging.debug("mAP({:.2f}, {}) = {:.2f}".format(overlap_thresh, map_type,
                                                    map_stat))
    return map_stat, xywh_res


def prune_zero_padding(gt_box, gt_label, difficult=None):
    valid_cnt = 0
    for i in range(len(gt_box)):
        if gt_box[i, 0] == 0 and gt_box[i, 1] == 0 and \
                gt_box[i, 2] == 0 and gt_box[i, 3] == 0:
            break
        valid_cnt += 1
    return (gt_box[:valid_cnt], gt_label[:valid_cnt], difficult[:valid_cnt]
            if difficult is not None else None)


def bbox_area(bbox, is_bbox_normalized):
    """
    Calculate area of a bounding box
    """
    norm = 1. - float(is_bbox_normalized)
    width = bbox[2] - bbox[0] + norm
    height = bbox[3] - bbox[1] + norm
    return width * height


def jaccard_overlap(pred, gt, is_bbox_normalized=False):
    """
    Calculate jaccard overlap ratio between two bounding box
    """
    if pred[0] >= gt[2] or pred[2] <= gt[0] or \
        pred[1] >= gt[3] or pred[3] <= gt[1]:
        return 0.
    inter_xmin = max(pred[0], gt[0])
    inter_ymin = max(pred[1], gt[1])
    inter_xmax = min(pred[2], gt[2])
    inter_ymax = min(pred[3], gt[3])
    inter_size = bbox_area([inter_xmin, inter_ymin, inter_xmax, inter_ymax],
                           is_bbox_normalized)
    pred_size = bbox_area(pred, is_bbox_normalized)
    gt_size = bbox_area(gt, is_bbox_normalized)
    overlap = float(inter_size) / (pred_size + gt_size - inter_size)
    return overlap


class DetectionMAP(object):
    """
    Calculate detection mean average precision.
    Currently support two types: 11point and integral

    Args:
        class_num (int): the class number.
        overlap_thresh (float): The threshold of overlap
            ratio between prediction bounding box and
            ground truth bounding box for deciding
            true/false positive. Default 0.5.
        map_type (str): calculation method of mean average
            precision, currently support '11point' and
            'integral'. Default '11point'.
        is_bbox_normalized (bool): whther bounding boxes
            is normalized to range[0, 1]. Default False.
        evaluate_difficult (bool): whether to evaluate
            difficult bounding boxes. Default False.
    """

    def __init__(self,
                 class_num,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False):
        self.class_num = class_num
        self.overlap_thresh = overlap_thresh
        assert map_type in ['11point', 'integral'], \
                "map_type currently only support '11point' "\
                "and 'integral'"
        self.map_type = map_type
        self.is_bbox_normalized = is_bbox_normalized
        self.evaluate_difficult = evaluate_difficult
        self.reset()

    def update(self, bbox, gt_box, gt_label, difficult=None):
        """
        Update metric statics from given prediction and ground
        truth infomations.
        """
        if difficult is None:
            difficult = np.zeros_like(gt_label)

        # record class gt count
        for gtl, diff in zip(gt_label, difficult):
            if self.evaluate_difficult or int(diff) == 0:
                self.class_gt_counts[int(np.array(gtl))] += 1

        # record class score positive
        visited = [False] * len(gt_label)
        for b in bbox:
            label, score, xmin, ymin, xmax, ymax = b.tolist()
            pred = [xmin, ymin, xmax, ymax]
            max_idx = -1
            max_overlap = -1.0
            for i, gl in enumerate(gt_label):
                if int(gl) == int(label):
                    overlap = jaccard_overlap(pred, gt_box[i],
                                              self.is_bbox_normalized)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_idx = i

            if max_overlap > self.overlap_thresh:
                if self.evaluate_difficult or \
                        int(np.array(difficult[max_idx])) == 0:
                    if not visited[max_idx]:
                        self.class_score_poss[int(label)].append([score, 1.0])
                        visited[max_idx] = True
                    else:
                        self.class_score_poss[int(label)].append([score, 0.0])
            else:
                self.class_score_poss[int(label)].append([score, 0.0])

    def reset(self):
        """
        Reset metric statics
        """
        self.class_score_poss = [[] for _ in range(self.class_num)]
        self.class_gt_counts = [0] * self.class_num
        self.mAP = None
        self.APs = [None] * self.class_num

    def accumulate(self):
        """
        Accumulate metric results and calculate mAP
        """
        mAP = 0.
        valid_cnt = 0
        for id, (
                score_pos, count
        ) in enumerate(zip(self.class_score_poss, self.class_gt_counts)):
            if count == 0: continue
            if len(score_pos) == 0:
                valid_cnt += 1
                continue

            accum_tp_list, accum_fp_list = \
                    self._get_tp_fp_accum(score_pos)
            precision = []
            recall = []
            for ac_tp, ac_fp in zip(accum_tp_list, accum_fp_list):
                precision.append(float(ac_tp) / (ac_tp + ac_fp))
                recall.append(float(ac_tp) / count)

            if self.map_type == '11point':
                max_precisions = [0.] * 11
                start_idx = len(precision) - 1
                for j in range(10, -1, -1):
                    for i in range(start_idx, -1, -1):
                        if recall[i] < float(j) / 10.:
                            start_idx = i
                            if j > 0:
                                max_precisions[j - 1] = max_precisions[j]
                                break
                        else:
                            if max_precisions[j] < precision[i]:
                                max_precisions[j] = precision[i]
                mAP += sum(max_precisions) / 11.
                self.APs[id] = sum(max_precisions) / 11.
                valid_cnt += 1
            elif self.map_type == 'integral':
                import math
                ap = 0.
                prev_recall = 0.
                for i in range(len(precision)):
                    recall_gap = math.fabs(recall[i] - prev_recall)
                    if recall_gap > 1e-6:
                        ap += precision[i] * recall_gap
                        prev_recall = recall[i]
                mAP += ap
                self.APs[id] = sum(max_precisions) / 11.
                valid_cnt += 1
            else:
                raise Exception("Unspported mAP type {}".format(self.map_type))

        self.mAP = mAP / float(valid_cnt) if valid_cnt > 0 else mAP

    def get_map(self):
        """
        Get mAP result
        """
        if self.mAP is None:
            raise Exception("mAP is not calculated.")
        return self.mAP

    def _get_tp_fp_accum(self, score_pos_list):
        """
        Calculate accumulating true/false positive results from
        [score, pos] records
        """
        sorted_list = sorted(score_pos_list, key=lambda s: s[0], reverse=True)
        accum_tp = 0
        accum_fp = 0
        accum_tp_list = []
        accum_fp_list = []
        for (score, pos) in sorted_list:
            accum_tp += int(pos)
            accum_tp_list.append(accum_tp)
            accum_fp += 1 - int(pos)
            accum_fp_list.append(accum_fp)
        return accum_tp_list, accum_fp_list


def makeplot(rs, ps, outDir, class_name, iou_type):
    """针对某个特定类别，绘制不同评估要求下的准确率和召回率。
       绘制结果说明参考COCODataset官网给出分析工具说明https://cocodataset.org/#detection-eval。

       Refer to https://github.com/open-mmlab/mmdetection/blob/master/tools/coco_error_analysis.py

       Args:
           rs (np.array): 在不同置信度阈值下计算得到的召回率。
           ps (np.array): 在不同置信度阈值下计算得到的准确率。ps与rs相同位置下的数值为同一个置信度阈值
               计算得到的准确率与召回率。
           outDir (str): 图表保存的路径。
           class_name (str): 类别名。
           iou_type (str): iou计算方式，若为检测框，则设置为'bbox'，若为像素级分割结果，则设置为'segm'。

    """

    import matplotlib.pyplot as plt

    cs = np.vstack([
        np.ones((2, 3)), np.array([.31, .51, .74]), np.array([.75, .31, .30]),
        np.array([.36, .90, .38]), np.array([.50, .39, .64]),
        np.array([1, .6, 0])
    ])
    areaNames = ['allarea', 'small', 'medium', 'large']
    types = ['C75', 'C50', 'Loc', 'Sim', 'Oth', 'BG', 'FN']
    for i in range(len(areaNames)):
        area_ps = ps[..., i, 0]
        figure_tile = iou_type + '-' + class_name + '-' + areaNames[i]
        aps = [ps_.mean() for ps_ in area_ps]
        ps_curve = [
            ps_.mean(axis=1) if ps_.ndim > 1 else ps_ for ps_ in area_ps
        ]
        ps_curve.insert(0, np.zeros(ps_curve[0].shape))
        fig = plt.figure()
        ax = plt.subplot(111)
        for k in range(len(types)):
            ax.plot(rs, ps_curve[k + 1], color=[0, 0, 0], linewidth=0.5)
            ax.fill_between(
                rs,
                ps_curve[k],
                ps_curve[k + 1],
                color=cs[k],
                label=str('[{:.3f}'.format(aps[k]) + ']' + types[k]))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1.)
        plt.ylim(0, 1.)
        plt.title(figure_tile)
        plt.legend()
        fig.savefig(outDir + '/{}.png'.format(figure_tile))
        plt.close(fig)


def analyze_individual_category(k, cocoDt, cocoGt, catId, iou_type):
    """针对某个特定类别，分析忽略亚类混淆和类别混淆时的准确率。

       Refer to https://github.com/open-mmlab/mmdetection/blob/master/tools/coco_error_analysis.py

       Args:
           k (int): 待分析类别的序号。
           cocoDt (pycocotols.coco.COCO): 按COCO类存放的预测结果。
           cocoGt (pycocotols.coco.COCO): 按COCO类存放的真值。
           catId (int): 待分析类别在数据集中的类别id。
           iou_type (str): iou计算方式，若为检测框，则设置为'bbox'，若为像素级分割结果，则设置为'segm'。

       Returns:
           int:
           dict: 有关键字'ps_supercategory'和'ps_allcategory'。关键字'ps_supercategory'的键值是忽略亚类间
               混淆时的准确率，关键字'ps_allcategory'的键值是忽略类别间混淆时的准确率。

    """

    # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    # or matplotlib.backends is imported for the first time
    # pycocotools import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    nm = cocoGt.loadCats(catId)[0]
    logging.info('--------------analyzing {}-{}---------------'.format(
        k + 1, nm['name']))
    ps_ = {}
    dt = copy.deepcopy(cocoDt)
    nm = cocoGt.loadCats(catId)[0]
    imgIds = cocoGt.getImgIds()
    dt_anns = dt.dataset['annotations']
    select_dt_anns = []
    for ann in dt_anns:
        if ann['category_id'] == catId:
            select_dt_anns.append(ann)
    dt.dataset['annotations'] = select_dt_anns
    dt.createIndex()
    # compute precision but ignore superclass confusion
    gt = copy.deepcopy(cocoGt)
    child_catIds = gt.getCatIds(supNms=[nm['supercategory']])
    for idx, ann in enumerate(gt.dataset['annotations']):
        if (ann['category_id'] in child_catIds and
                ann['category_id'] != catId):
            gt.dataset['annotations'][idx]['ignore'] = 1
            gt.dataset['annotations'][idx]['iscrowd'] = 1
            gt.dataset['annotations'][idx]['category_id'] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [100]
    cocoEval.params.iouThrs = [.1]
    cocoEval.params.useCats = 1
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_supercategory = cocoEval.eval['precision'][0, :, k, :, :]
    ps_['ps_supercategory'] = ps_supercategory
    # compute precision but ignore any class confusion
    gt = copy.deepcopy(cocoGt)
    for idx, ann in enumerate(gt.dataset['annotations']):
        if ann['category_id'] != catId:
            gt.dataset['annotations'][idx]['ignore'] = 1
            gt.dataset['annotations'][idx]['iscrowd'] = 1
            gt.dataset['annotations'][idx]['category_id'] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [100]
    cocoEval.params.iouThrs = [.1]
    cocoEval.params.useCats = 1
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_allcategory = cocoEval.eval['precision'][0, :, k, :, :]
    ps_['ps_allcategory'] = ps_allcategory
    return k, ps_


def coco_error_analysis(eval_details_file=None,
                        gt=None,
                        pred_bbox=None,
                        pred_mask=None,
                        save_dir='./output'):
    """逐个分析模型预测错误的原因，并将分析结果以图表的形式展示。
       分析结果说明参考COCODataset官网给出分析工具说明https://cocodataset.org/#detection-eval。

       Refer to https://github.com/open-mmlab/mmdetection/blob/master/tools/coco_error_analysis.py

       Args:
           eval_details_file (str):  模型评估结果的保存路径，包含真值信息和预测结果。
           gt (list): 数据集的真值信息。默认值为None。
           pred_bbox (list): 模型在数据集上的预测框。默认值为None。
           pred_mask (list): 模型在数据集上的预测mask。默认值为None。
           save_dir (str): 可视化结果保存路径。默认值为'./output'。

        Note:
           eval_details_file的优先级更高，只要eval_details_file不为None，
           就会从eval_details_file提取真值信息和预测结果做分析。
           当eval_details_file为None时，则用gt、pred_mask、pred_mask做分析。

    """

    import multiprocessing as mp
    # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    # or matplotlib.backends is imported for the first time
    # pycocotools import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if eval_details_file is not None:
        import json
        with open(eval_details_file, 'r') as f:
            eval_details = json.load(f)
            pred_bbox = eval_details['bbox']
            if 'mask' in eval_details:
                pred_mask = eval_details['mask']
            gt = eval_details['gt']
    if gt is None or pred_bbox is None:
        raise Exception(
            "gt/pred_bbox/pred_mask is None now, please set right eval_details_file or gt/pred_bbox/pred_mask."
        )
    if pred_bbox is not None and len(pred_bbox) == 0:
        raise Exception("There is no predicted bbox.")
    if pred_mask is not None and len(pred_mask) == 0:
        raise Exception("There is no predicted mask.")

    def _analyze_results(cocoGt, cocoDt, res_type, out_dir):
        directory = os.path.dirname(out_dir + '/')
        if not os.path.exists(directory):
            logging.info('-------------create {}-----------------'.format(
                out_dir))
            os.makedirs(directory)

        imgIds = cocoGt.getImgIds()
        res_out_dir = out_dir + '/' + res_type + '/'
        res_directory = os.path.dirname(res_out_dir)
        if not os.path.exists(res_directory):
            logging.info('-------------create {}-----------------'.format(
                res_out_dir))
            os.makedirs(res_directory)
        iou_type = res_type
        cocoEval = COCOeval(
            copy.deepcopy(cocoGt), copy.deepcopy(cocoDt), iou_type)
        cocoEval.params.imgIds = imgIds
        cocoEval.params.iouThrs = [.75, .5, .1]
        cocoEval.params.maxDets = [100]
        cocoEval.evaluate()
        cocoEval.accumulate()
        ps = cocoEval.eval['precision']
        ps = np.vstack([ps, np.zeros((4, *ps.shape[1:]))])
        catIds = cocoGt.getCatIds()
        recThrs = cocoEval.params.recThrs
        thread_num = mp.cpu_count() if mp.cpu_count() < 8 else 8
        thread_pool = mp.pool.ThreadPool(thread_num)
        args = [(k, cocoDt, cocoGt, catId, iou_type)
                for k, catId in enumerate(catIds)]
        analyze_results = thread_pool.starmap(analyze_individual_category,
                                              args)
        for k, catId in enumerate(catIds):
            nm = cocoGt.loadCats(catId)[0]
            logging.info('--------------saving {}-{}---------------'.format(
                k + 1, nm['name']))
            analyze_result = analyze_results[k]
            assert k == analyze_result[0], ""
            ps_supercategory = analyze_result[1]['ps_supercategory']
            ps_allcategory = analyze_result[1]['ps_allcategory']
            # compute precision but ignore superclass confusion
            ps[3, :, k, :, :] = ps_supercategory
            # compute precision but ignore any class confusion
            ps[4, :, k, :, :] = ps_allcategory
            # fill in background and false negative errors and plot
            T, _, _, A, _ = ps.shape
            for t in range(T):
                for a in range(A):
                    if np.sum(ps[t, :, k, a, :] ==
                              -1) != len(ps[t, :, k, :, :]):
                        ps[t, :, k, a, :][ps[t, :, k, a, :] == -1] = 0
            ps[5, :, k, :, :] = (ps[4, :, k, :, :] > 0)
            ps[6, :, k, :, :] = 1.0
            makeplot(recThrs, ps[:, :, k], res_out_dir, nm['name'], iou_type)
        makeplot(recThrs, ps, res_out_dir, 'allclass', iou_type)

    np.linspace = fixed_linspace
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
    # or matplotlib.backends is imported for the first time
    # pycocotools import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    from pycocotools.cocoeval import COCOeval
    if pred_bbox is not None:
        coco_dt = loadRes(coco_gt, pred_bbox)
        _analyze_results(coco_gt, coco_dt, res_type='bbox', out_dir=save_dir)
    if pred_mask is not None:
        coco_dt = loadRes(coco_gt, pred_mask)
        _analyze_results(coco_gt, coco_dt, res_type='segm', out_dir=save_dir)
    np.linspace = backup_linspace
    logging.info("The analysis figures are saved in {}".format(save_dir))
