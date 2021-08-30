# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import division
from __future__ import print_function

import sys
import copy
import os
import os.path as osp
import numpy as np
import itertools
from paddlex.ppdet.metrics.map_utils import draw_pr_curve
from paddlex.ppdet.metrics.json_results import get_det_res, get_det_poly_res, get_seg_res, get_solov2_segm_res
import paddlex.utils.logging as logging


def get_infer_results(outs, catid, bias=0):
    """
    Get result at the stage of inference.
    The output format is dictionary containing bbox or mask result.

    For example, bbox result is a list and each element contains
    image_id, category_id, bbox and score.
    """
    if outs is None or len(outs) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.'
        )

    im_id = outs['im_id']

    infer_res = {}
    if 'bbox' in outs:
        if len(outs['bbox']) > 0 and len(outs['bbox'][0]) > 6:
            infer_res['bbox'] = get_det_poly_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)
        else:
            infer_res['bbox'] = get_det_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)

    if 'mask' in outs:
        # mask post process
        infer_res['mask'] = get_seg_res(outs['mask'], outs['bbox'],
                                        outs['bbox_num'], im_id, catid)

    if 'segm' in outs:
        infer_res['segm'] = get_solov2_segm_res(outs, im_id, catid)

    return infer_res


def cocoapi_eval(anns,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000),
                 classwise=False):
    """
    Args:
        anns: Evaluation result.
        style (str): COCOeval style, can be `bbox` , `segm` and `proposal`.
        coco_gt (str): Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file (str): COCO annotations file.
        max_dets (tuple): COCO evaluation maxDets.
        classwise (bool): Whether per-category AP and draw P-R Curve or not.
    """
    assert coco_gt is not None or anno_file is not None
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if coco_gt is None:
        coco_gt = COCO(anno_file)
    logging.info("Start evaluate...")
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
    if classwise:
        # Compute per-category AP and PR curve
        try:
            from terminaltables import AsciiTable
        except Exception as e:
            logging.error(
                'terminaltables not found, plaese install terminaltables. '
                'for example: `pip install terminaltables`.')
            raise e
        precisions = coco_eval.eval['precision']
        cat_ids = coco_gt.getCatIds()
        # precision: (iou, recall, cls, area range, max dets)
        assert len(cat_ids) == precisions.shape[2]
        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco_gt.loadCats(catId)[0]
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (str(nm["name"]), '{:0.3f}'.format(float(ap))))
            pr_array = precisions[0, :, idx, 0, 2]
            recall_array = np.arange(0.0, 1.01, 0.01)
            draw_pr_curve(
                pr_array,
                recall_array,
                out_dir=style + '_pr_curve',
                file_name='{}_precision_recall_curve.jpg'.format(nm["name"]))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        logging.info('Per-category of {} AP: \n{}'.format(style, table.table))
        logging.info("per-category PR curve has output to {} folder.".format(
            style + '_pr_curve'))
    # flush coco evaluation result
    sys.stdout.flush()
    return coco_eval.stats


def loadRes(coco_obj, anns):
    """
    Load result file and return a result api object.
    :param   resFile (str)     : file name of result file
    :return: res (obj)         : result api object
    """

    # This function has the same functionality as pycocotools.COCO.loadRes,
    # except that the input anns is list of results rather than a json file.
    # Refer to
    # https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/coco.py#L305,

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


def makeplot(rs, ps, outDir, class_name, iou_type):
    import matplotlib.pyplot as plt
    cs = np.vstack([
        np.ones((2, 3)),
        np.array([0.31, 0.51, 0.74]),
        np.array([0.75, 0.31, 0.30]),
        np.array([0.36, 0.90, 0.38]),
        np.array([0.50, 0.39, 0.64]),
        np.array([1, 0.6, 0]),
    ])
    areaNames = ['allarea', 'small', 'medium', 'large']
    types = ['C75', 'C50', 'Loc', 'Sim', 'Oth', 'BG', 'FN']
    for i in range(len(areaNames)):
        area_ps = ps[..., i, 0]
        figure_title = iou_type + '-' + class_name + '-' + areaNames[i]
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
                label=str(f'[{aps[k]:.3f}]' + types[k]), )
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        plt.title(figure_title)
        plt.legend()
        # plt.show()
        fig.savefig(osp.join(outDir, f'{figure_title}.png'))
        plt.close(fig)


def analyze_individual_category(k, cocoDt, cocoGt, catId, iou_type,
                                areas=None):
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
    print(f'--------------analyzing {k + 1}-{nm["name"]}---------------')
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
        if ann['category_id'] in child_catIds and ann['category_id'] != catId:
            gt.dataset['annotations'][idx]['ignore'] = 1
            gt.dataset['annotations'][idx]['iscrowd'] = 1
            gt.dataset['annotations'][idx]['category_id'] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [100]
    cocoEval.params.iouThrs = [0.1]
    cocoEval.params.useCats = 1
    if areas:
        cocoEval.params.areaRng = [[0**2, areas[2]], [0**2, areas[0]],
                                   [areas[0], areas[1]], [areas[1], areas[2]]]
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
    cocoEval.params.iouThrs = [0.1]
    cocoEval.params.useCats = 1
    if areas:
        cocoEval.params.areaRng = [[0**2, areas[2]], [0**2, areas[0]],
                                   [areas[0], areas[1]], [areas[1], areas[2]]]
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

       Refer to https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/coco_error_analysis.py

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
        directory = osp.dirname(osp.join(out_dir, ''))
        if not osp.exists(directory):
            logging.info('-------------create {}-----------------'.format(
                out_dir))
            os.makedirs(directory)

        imgIds = cocoGt.getImgIds()
        res_out_dir = osp.join(out_dir, res_type, '')
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
            ps[ps == -1] = 0
            ps[5, :, k, :, :] = ps[4, :, k, :, :] > 0
            ps[6, :, k, :, :] = 1.0
            makeplot(recThrs, ps[:, :, k], res_out_dir, nm['name'], iou_type)
        makeplot(recThrs, ps, res_out_dir, 'allclass', iou_type)

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()

    if pred_bbox is not None:
        coco_dt = loadRes(coco_gt, pred_bbox)
        _analyze_results(coco_gt, coco_dt, res_type='bbox', out_dir=save_dir)
    if pred_mask is not None:
        coco_dt = loadRes(coco_gt, pred_mask)
        _analyze_results(coco_gt, coco_dt, res_type='segm', out_dir=save_dir)
    logging.info("The analysis figures are saved in {}".format(save_dir))
