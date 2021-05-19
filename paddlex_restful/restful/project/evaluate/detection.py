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

import json
import yaml
import copy
import os.path as osp
import numpy as np

backup_linspace = np.linspace


def fixed_linspace(start,
                   stop,
                   num=50,
                   endpoint=True,
                   retstep=False,
                   dtype=None,
                   axis=0):
    '''解决numpy > 1.17.2时pycocotools中linspace的报错问题。
    '''
    num = int(num)
    return backup_linspace(start, stop, num, endpoint, retstep, dtype, axis)


def jaccard_overlap(pred, gt):
    '''计算两个框之间的IoU。
    '''

    def bbox_area(bbox):
        width = bbox[2] - bbox[0] + 1
        height = bbox[3] - bbox[1] + 1
        return width * height
    if pred[0] >= gt[2] or pred[2] <= gt[0] or \
        pred[1] >= gt[3] or pred[3] <= gt[1]:
        return 0.
    inter_xmin = max(pred[0], gt[0])
    inter_ymin = max(pred[1], gt[1])
    inter_xmax = min(pred[2], gt[2])
    inter_ymax = min(pred[3], gt[3])
    inter_size = bbox_area([inter_xmin, inter_ymin, inter_xmax, inter_ymax])
    pred_size = bbox_area(pred)
    gt_size = bbox_area(gt)
    overlap = float(inter_size) / (pred_size + gt_size - inter_size)
    return overlap


def loadRes(coco_obj, anns):
    '''导入结果文件并返回pycocotools中的COCO对象。
    '''
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
    if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
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
            ann['area'] = maskUtils.area(ann['segmentation'])
            if not 'bbox' in ann:
                ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
            ann['id'] = id + 1
            ann['iscrowd'] = 0
    res.dataset['annotations'] = anns
    res.createIndex()
    return res


class DetectionMAP(object):
    def __init__(self,
                 num_classes,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False):
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        assert map_type in ['11point', 'integral'], \
                "map_type currently only support '11point' "\
                "and 'integral'"
        self.map_type = map_type
        self.is_bbox_normalized = is_bbox_normalized
        self.evaluate_difficult = evaluate_difficult
        self.reset()

    def update(self, bbox, gt_box, gt_label, difficult=None):
        '''用预测值和真值更新指标。
        '''
        if difficult is None:
            difficult = np.zeros_like(gt_label)

        for gtl, diff in zip(gt_label, difficult):
            if self.evaluate_difficult or int(diff) == 0:
                self.class_gt_counts[int(np.array(gtl))] += 1

        visited = [False] * len(gt_label)
        for b in bbox:
            label, score, xmin, ymin, xmax, ymax = b.tolist()
            pred = [xmin, ymin, xmax, ymax]
            max_idx = -1
            max_overlap = -1.0
            for i, gl in enumerate(gt_label):
                if int(gl) == int(label):
                    overlap = jaccard_overlap(pred, gt_box[i])
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
        '''初始化指标。
        '''
        self.class_score_poss = [[] for _ in range(self.num_classes)]
        self.class_gt_counts = [0] * self.num_classes
        self.mAP = None
        self.APs = [None] * self.num_classes

    def accumulate(self):
        '''汇总指标并由此计算mAP。
        '''
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
        '''获取mAP。
        '''
        if self.mAP is None:
            raise Exception("mAP is not calculated.")
        return self.mAP

    def _get_tp_fp_accum(self, score_pos_list):
        '''计算真阳/假阳。
        '''
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


class DetConfusionMatrix(object):
    def __init__(self,
                 num_classes,
                 overlap_thresh=0.5,
                 evaluate_difficult=False,
                 score_threshold=0.3):
        self.overlap_thresh = overlap_thresh
        self.evaluate_difficult = evaluate_difficult
        self.confusion_matrix = np.zeros(shape=(num_classes, num_classes))
        self.score_threshold = score_threshold
        self.total_tp = [0] * num_classes
        self.total_gt = [0] * num_classes
        self.total_pred = [0] * num_classes

    def update(self, bbox, gt_box, gt_label, difficult=None):
        '''更新混淆矩阵。
        '''
        if difficult is None:
            difficult = np.zeros_like(gt_label)

        dtind = np.argsort([-d[1] for d in bbox], kind='mergesort')
        bbox = [bbox[i] for i in dtind]
        det_bbox = []
        det_label = []
        G = len(gt_box)
        D = len(bbox)
        gtm = np.full((G, ), -1)
        dtm = np.full((D, ), -1)
        for j, b in enumerate(bbox):
            label, score, xmin, ymin, xmax, ymax = b.tolist()
            if float(score) < self.score_threshold:
                continue
            det_label.append(int(label) - 1)
            self.total_pred[int(label) - 1] += 1
            det_bbox.append([xmin, ymin, xmax, ymax])
        for i, gl in enumerate(gt_label):
            self.total_gt[int(gl) - 1] += 1

        for j, pred in enumerate(det_bbox):
            m = -1
            for i, gt in enumerate(gt_box):
                overlap = jaccard_overlap(pred, gt)
                if overlap >= self.overlap_thresh:
                    m = i
            if m == -1:
                continue
            gtm[m] = j
            dtm[j] = m
        for i, gl in enumerate(gt_label):
            if gtm[i] == -1:
                self.confusion_matrix[int(gl) - 1][self.confusion_matrix.shape[
                    1] - 1] += 1
        for i, b in enumerate(det_bbox):
            if dtm[i] > -1:
                gl = int(gt_label[dtm[i]]) - 1
                self.confusion_matrix[gl][int(det_label[i])] += 1
            if dtm[i] == -1:
                self.confusion_matrix[self.confusion_matrix.shape[0] - 1][int(
                    det_label[i])] += 1

        gtm = np.full((G, ), -1)
        dtm = np.full((D, ), -1)
        for j, pred in enumerate(det_bbox):
            m = -1
            max_overlap = -1
            for i, gt in enumerate(gt_box):
                if int(gt_label[i]) - 1 == int(det_label[j]):
                    overlap = jaccard_overlap(pred, gt)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        m = i
            if max_overlap < self.overlap_thresh:
                continue
            if difficult[m]:
                continue
            if m == -1 or gtm[m] > -1:
                continue
            gtm[m] = j
            dtm[j] = m
            self.total_tp[int(gt_label[m]) - 1] += 1

    def get_confusion_matrix(self):
        return self.confusion_matrix


class InsSegConfusionMatrix(object):
    def __init__(self,
                 num_classes,
                 overlap_thresh=0.5,
                 evaluate_difficult=False,
                 score_threshold=0.3):
        self.overlap_thresh = overlap_thresh
        self.evaluate_difficult = evaluate_difficult
        self.confusion_matrix = np.zeros(shape=(num_classes, num_classes))
        self.score_threshold = score_threshold
        self.total_tp = [0] * num_classes
        self.total_gt = [0] * num_classes
        self.total_pred = [0] * num_classes

    def update(self, mask, gt_mask, gt_label, is_crowd=None):
        '''更新混淆矩阵。
        '''
        dtind = np.argsort([-d[1] for d in mask], kind='mergesort')
        mask = [mask[i] for i in dtind]
        det_mask = []
        det_label = []
        for j, b in enumerate(mask):
            label, score, d_b = b
            if float(score) < self.score_threshold:
                continue
            self.total_pred[int(label) - 1] += 1
            det_label.append(label - 1)
            det_mask.append(d_b)
        for i, gl in enumerate(gt_label):
            self.total_gt[int(gl) - 1] += 1

        g = [gt for gt in gt_mask]
        d = [dt for dt in det_mask]
        import pycocotools.mask as maskUtils
        ious = maskUtils.iou(d, g, is_crowd)
        G = len(gt_mask)
        D = len(det_mask)
        gtm = np.full((G, ), -1)
        dtm = np.full((D, ), -1)
        gtIg = np.array(is_crowd)
        dtIg = np.zeros((D, ))
        for dind, d in enumerate(det_mask):
            m = -1
            for gind, g in enumerate(gt_mask):
                if ious[dind, gind] >= self.overlap_thresh:
                    m = gind
            if m == -1:
                continue
            dtIg[dind] = gtIg[m]
            dtm[dind] = m
            gtm[m] = dind
        for i, gl in enumerate(gt_label):
            if gtm[i] == -1 and gtIg[i] == 0:
                self.confusion_matrix[int(gl) - 1][self.confusion_matrix.shape[
                    1] - 1] += 1
        for i, b in enumerate(det_mask):
            if dtm[i] > -1 and dtIg[i] == 0:
                gl = int(gt_label[dtm[i]]) - 1
                self.confusion_matrix[gl][int(det_label[i])] += 1
            if dtm[i] == -1 and dtIg[i] == 0:
                self.confusion_matrix[self.confusion_matrix.shape[0] - 1][int(
                    det_label[i])] += 1

        gtm = np.full((G, ), -1)
        dtm = np.full((D, ), -1)
        for dind, d in enumerate(det_mask):
            m = -1
            max_overlap = -1
            for gind, g in enumerate(gt_mask):
                if int(gt_label[gind]) - 1 == int(det_label[dind]):
                    if ious[dind, gind] > max_overlap:
                        max_overlap = ious[dind, gind]
                        m = gind

            if max_overlap < self.overlap_thresh:
                continue
            if m == -1 or gtm[m] > -1:
                continue
            dtm[dind] = m
            gtm[m] = dind
            self.total_tp[int(gt_label[m]) - 1] += 1

    def get_confusion_matrix(self):
        return self.confusion_matrix


class DetEvaluator(object):
    def __init__(self, model_path, overlap_thresh=0.5, score_threshold=0.3):
        self.model_path = model_path
        self.overlap_thresh = overlap_thresh
        self.score_threshold = score_threshold

    def _prepare_data(self):
        with open(osp.join(self.model_path, 'eval_details.json'), 'r') as f:
            eval_details = json.load(f)
        self.bbox = eval_details['bbox']
        self.mask = None
        if 'mask' in eval_details:
            self.mask = eval_details['mask']
        gt_dataset = eval_details['gt']

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        self.coco = COCO()
        self.coco.dataset = gt_dataset
        self.coco.createIndex()
        img_ids = self.coco.getImgIds()
        cat_ids = self.coco.getCatIds()
        self.catid2clsid = dict(
            {catid: i + 1
             for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            self.coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })
        self.cid2cname = dict(
            {cid: cname
             for cname, cid in self.cname2cid.items()})
        self.cid2cname[0] = 'back_ground'

        self.gt = dict()
        for img_id in img_ids:
            img_anno = self.coco.loadImgs(img_id)[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            ins_anno_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            instances = self.coco.loadAnns(ins_anno_ids)

            bboxes = []
            for inst in instances:
                x, y, box_w, box_h = inst['bbox']
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(im_w - 1, x1 + max(0, box_w - 1))
                y2 = min(im_h - 1, y1 + max(0, box_h - 1))
                if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                    inst['clean_bbox'] = [x1, y1, x2, y2]
                    bboxes.append(inst)
                else:
                    pass
            num_bbox = len(bboxes)

            gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
            gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
            gt_score = np.ones((num_bbox, 1), dtype=np.float32)
            is_crowd = np.zeros((num_bbox), dtype=np.int32)
            difficult = np.zeros((num_bbox, 1), dtype=np.int32)
            gt_poly = [None] * num_bbox

            for i, box in enumerate(bboxes):
                catid = box['category_id']
                gt_class[i][0] = self.catid2clsid[catid]
                gt_bbox[i, :] = box['clean_bbox']
                is_crowd[i] = box['iscrowd']
                if 'segmentation' in box:
                    gt_poly[i] = self.coco.annToRLE(box)
                if 'difficult' in box:
                    difficult[i][0] = box['difficult']

            coco_rec = {
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_bbox': gt_bbox,
                'gt_score': gt_score,
                'gt_poly': gt_poly,
                'difficult': difficult
            }
            self.gt[img_id] = coco_rec
        self.gtimgids = list(self.gt.keys())
        self.detimgids = [ann['image_id'] for ann in self.bbox]
        self.det = dict()
        if len(self.bbox) > 0:
            if 'bbox' in self.bbox[0] and not self.bbox[0]['bbox'] == []:
                for id, ann in enumerate(self.bbox):
                    im_id = ann['image_id']
                    bb = ann['bbox']
                    x1, x2, y1, y2 = [
                        bb[0], bb[0] + bb[2] - 1, bb[1], bb[1] + bb[3] - 1
                    ]
                    score = ann['score']
                    category_id = self.catid2clsid[ann['category_id']]
                    if int(im_id) not in self.det:
                        self.det[int(im_id)] = [[
                            category_id, score, x1, y1, x2, y2
                        ]]
                    else:
                        self.det[int(im_id)].extend(
                            [[category_id, score, x1, y1, x2, y2]])

        if self.mask is not None:
            self.maskimgids = [ann['image_id'] for ann in self.mask]
            self.segm = dict()
            if len(self.mask) > 0:
                if 'segmentation' in self.mask[0]:
                    for id, ann in enumerate(self.mask):
                        im_id = ann['image_id']
                        score = ann['score']
                        segmentation = self.coco.annToRLE(ann)
                        category_id = self.catid2clsid[ann['category_id']]
                        if int(im_id) not in self.segm:
                            self.segm[int(im_id)] = [[
                                category_id, score, segmentation
                            ]]
                        else:
                            self.segm[int(im_id)].extend(
                                [[category_id, score, segmentation]])

    def cal_confusion_matrix(self):
        '''计算混淆矩阵。
        '''
        self._prepare_data()
        confusion_matrix = DetConfusionMatrix(
            num_classes=len(self.cid2cname.keys()),
            overlap_thresh=self.overlap_thresh,
            score_threshold=self.score_threshold)
        for im_id in self.gtimgids:
            if im_id not in set(self.detimgids):
                bbox = []
            else:
                bbox = np.array(self.det[im_id])
            gt_box = self.gt[im_id]['gt_bbox']
            gt_label = self.gt[im_id]['gt_class']
            difficult = self.gt[im_id]['difficult']
            confusion_matrix.update(bbox, gt_box, gt_label, difficult)
        self.confusion_matrix = confusion_matrix.get_confusion_matrix()

        self.precision_recall = dict()
        for id in range(len(self.cid2cname.keys()) - 1):
            if confusion_matrix.total_gt[id] == 0:
                recall = -1
            else:
                recall = confusion_matrix.total_tp[
                    id] / confusion_matrix.total_gt[id]
            if confusion_matrix.total_pred[id] == 0:
                precision = -1
            else:
                precision = confusion_matrix.total_tp[
                    id] / confusion_matrix.total_pred[id]
            self.precision_recall[self.cid2cname[id + 1]] = {
                "precision": precision,
                "recall": recall
            }
        return self.confusion_matrix

    def cal_precision_recall(self):
        '''计算precision、recall。
        '''
        return self.precision_recall

    def cal_map(self):
        '''计算mAP。
        '''
        detection_map = DetectionMAP(
            num_classes=len(self.cid2cname.keys()),
            overlap_thresh=self.overlap_thresh)
        for im_id in self.gtimgids:
            if im_id not in set(self.detimgids):
                bbox = []
            else:
                bbox = np.array(self.det[im_id])
            gt_box = self.gt[im_id]['gt_bbox']
            gt_label = self.gt[im_id]['gt_class']
            difficult = self.gt[im_id]['difficult']
            detection_map.update(bbox, gt_box, gt_label, difficult)
        detection_map.accumulate()
        self.map = detection_map.get_map()
        self.APs = detection_map.APs
        return self.map

    def cal_ap(self):
        '''计算各类AP。
        '''
        self.aps = dict()
        for id, ap in enumerate(self.APs):
            if id == 0:
                continue
            self.aps[self.cid2cname[id]] = ap
        return self.aps

    def generate_report(self):
        '''生成评估报告。
        '''
        report = dict()
        report['Confusion_Matrix'] = copy.deepcopy(self.cal_confusion_matrix())
        report['mAP'] = copy.deepcopy(self.cal_map())
        report['PRAP'] = copy.deepcopy(self.cal_precision_recall())
        report['label_list'] = copy.deepcopy(list(self.cname2cid.keys()))
        report['label_list'].append('back_ground')
        per_ap = copy.deepcopy(self.cal_ap())
        for k, v in per_ap.items():
            report['PRAP'][k]["AP"] = v
        return report


class InsSegEvaluator(DetEvaluator):
    def __init__(self, model_path, overlap_thresh=0.5, score_threshold=0.3):
        super(DetEvaluator, self).__init__()
        self.model_path = model_path
        self.overlap_thresh = overlap_thresh
        self.score_threshold = score_threshold

    def cal_confusion_matrix_mask(self):
        '''计算Mask的混淆矩阵。
        '''
        confusion_matrix = InsSegConfusionMatrix(
            num_classes=len(self.cid2cname.keys()),
            overlap_thresh=self.overlap_thresh,
            score_threshold=self.score_threshold)
        for im_id in self.gtimgids:
            if im_id not in set(self.maskimgids):
                segm = []
            else:
                segm = self.segm[im_id]
            gt_segm = self.gt[im_id]['gt_poly']
            gt_label = self.gt[im_id]['gt_class']
            is_crowd = self.gt[im_id]['is_crowd']
            confusion_matrix.update(segm, gt_segm, gt_label, is_crowd)
        self.confusion_matrix_mask = confusion_matrix.get_confusion_matrix()

        self.precision_recall_mask = dict()
        for id in range(len(self.cid2cname.keys()) - 1):
            if confusion_matrix.total_gt[id] == 0:
                recall = -1
            else:
                recall = confusion_matrix.total_tp[
                    id] / confusion_matrix.total_gt[id]
            if confusion_matrix.total_pred[id] == 0:
                precision = -1
            else:
                precision = confusion_matrix.total_tp[
                    id] / confusion_matrix.total_pred[id]
            self.precision_recall_mask[self.cid2cname[id + 1]] = {
                "precision": precision,
                "recall": recall
            }
        return self.confusion_matrix_mask

    def cal_precision_recall_mask(self):
        '''计算Mask的precision、recall。
        '''
        return self.precision_recall_mask

    def _summarize(self,
                   coco_gt,
                   ap=1,
                   iouThr=None,
                   areaRng='all',
                   maxDets=100):
        p = coco_gt.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            s = coco_gt.eval['precision']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            s = coco_gt.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def cal_map(self):
        '''计算BBox的mAP。
        '''
        if len(self.bbox) > 0:
            from pycocotools.cocoeval import COCOeval
            coco_dt = loadRes(self.coco, self.bbox)
            np.linspace = fixed_linspace
            coco_eval = COCOeval(self.coco, coco_dt, 'bbox')
            coco_eval.params.iouThrs = np.linspace(
                self.overlap_thresh, self.overlap_thresh, 1, endpoint=True)
            np.linspace = backup_linspace
            coco_eval.evaluate()
            coco_eval.accumulate()
            self.map = self._summarize(coco_eval, iouThr=self.overlap_thresh)

            precision = coco_eval.eval['precision'][0, :, :, 0, 2]
            num_classes = len(coco_eval.params.catIds)
            self.APs = [None] * num_classes
            for i in range(num_classes):
                per = precision[:, i]
                per = per[per > -1]
                self.APs[i] = np.sum(per) / 101 if per.shape[0] > 0 else None
        else:
            self.map = None
            self.APs = [None] * len(self.catid2clsid)
        return self.map

    def cal_ap(self):
        '''计算BBox的各类AP。
        '''
        self.aps = dict()
        for id, ap in enumerate(self.APs):
            self.aps[self.cid2cname[id + 1]] = ap
        return self.aps

    def cal_map_mask(self):
        '''计算Mask的mAP。
        '''
        if len(self.mask) > 0:
            from pycocotools.cocoeval import COCOeval
            coco_dt = loadRes(self.coco, self.mask)
            np.linspace = fixed_linspace
            coco_eval = COCOeval(self.coco, coco_dt, 'segm')
            coco_eval.params.iouThrs = np.linspace(
                self.overlap_thresh, self.overlap_thresh, 1, endpoint=True)
            np.linspace = backup_linspace
            coco_eval.evaluate()
            coco_eval.accumulate()
            self.map_mask = self._summarize(
                coco_eval, iouThr=self.overlap_thresh)

            precision = coco_eval.eval['precision'][0, :, :, 0, 2]
            num_classes = len(coco_eval.params.catIds)
            self.mask_APs = [None] * num_classes
            for i in range(num_classes):
                per = precision[:, i]
                per = per[per > -1]
                self.mask_APs[i] = np.sum(per) / 101 if per.shape[
                    0] > 0 else None
        else:
            self.map_mask = None
            self.mask_APs = [None] * len(self.catid2clsid)
        return self.map_mask

    def cal_ap_mask(self):
        '''计算Mask的各类AP。
        '''
        self.mask_aps = dict()
        for id, ap in enumerate(self.mask_APs):
            self.mask_aps[self.cid2cname[id + 1]] = ap
        return self.mask_aps

    def generate_report(self):
        '''生成评估报告。
        '''
        report = dict()
        report['BBox_Confusion_Matrix'] = copy.deepcopy(
            self.cal_confusion_matrix())
        report['BBox_mAP'] = copy.deepcopy(self.cal_map())
        report['BBox_PRAP'] = copy.deepcopy(self.cal_precision_recall())
        report['label_list'] = copy.deepcopy(list(self.cname2cid.keys()))
        report['label_list'].append('back_ground')
        per_ap = copy.deepcopy(self.cal_ap())
        for k, v in per_ap.items():
            report['BBox_PRAP'][k]['AP'] = v

        report['Mask_Confusion_Matrix'] = copy.deepcopy(
            self.cal_confusion_matrix_mask())
        report['Mask_mAP'] = copy.deepcopy(self.cal_map_mask())
        report['Mask_PRAP'] = copy.deepcopy(self.cal_precision_recall_mask())
        per_ap_mask = copy.deepcopy(self.cal_ap_mask())
        for k, v in per_ap_mask.items():
            report['Mask_PRAP'][k]['AP'] = v
        return report
