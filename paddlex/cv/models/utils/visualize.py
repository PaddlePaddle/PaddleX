#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import cv2
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from .detection_eval import fixed_linspace, backup_linspace, loadRes


def visualize_detection(image, result, threshold=0.5, save_dir=None):
    """
        Visualize bbox and mask results
    """

    image_name = os.path.split(image)[-1]
    image = Image.open(image).convert('RGB')
    image = draw_bbox_mask(image, result, threshold=threshold)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        out_path = os.path.join(save_dir, 'visualize_{}'.format(image_name))
        image.save(out_path, quality=95)
    else:
        return image


def visualize_segmentation(image, result, weight=0.6, save_dir=None):
    """
    Convert segment result to color image, and save added image.
    Args:
        image: the path of origin image
        result: the predict result of image
        weight: the image weight of visual image, and the result weight is (1 - weight)
        save_dir: the directory for saving visual image
    """
    label_map = result['label_map']
    color_map = get_color_map_list(256)
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(label_map, color_map[:, 0])
    c2 = cv2.LUT(label_map, color_map[:, 1])
    c3 = cv2.LUT(label_map, color_map[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))

    im = cv2.imread(image)
    vis_result = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, 'visualize_{}'.format(image_name))
        cv2.imwrite(out_path, vis_result)
    else:
        return vis_result


def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


# expand an array of boxes by a given scale.
def expand_boxes(boxes, scale):
    """
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


def clip_bbox(bbox):
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax


def draw_bbox_mask(image, results, threshold=0.5, alpha=0.7):
    labels = list()
    for dt in np.array(results):
        if dt['category'] not in labels:
            labels.append(dt['category'])
    color_map = get_color_map_list(len(labels))

    for dt in np.array(results):
        cname, bbox, score = dt['category'], dt['bbox'], dt['score']
        if score < threshold:
            continue

        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h

        color = tuple(color_map[labels.index(cname)])

        # draw bbox
        draw = ImageDraw.Draw(image)
        draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                   (xmin, ymin)],
                  width=2,
                  fill=color)

        # draw label
        text = "{} {:.2f}".format(cname, score)
        tw, th = draw.textsize(text)
        draw.rectangle([(xmin + 1, ymin - th), (xmin + tw + 1, ymin)],
                       fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

        # draw mask
        if 'mask' in dt:
            mask = dt['mask']
            color_mask = np.array(color_map[labels.index(
                dt['category'])]).astype('float32')
            img_array = np.array(image).astype('float32')
            idx = np.nonzero(mask)
            img_array[idx[0], idx[1], :] *= 1.0 - alpha
            img_array[idx[0], idx[1], :] += alpha * color_mask
            image = Image.fromarray(img_array.astype('uint8'))
    return image


def draw_pr_curve(eval_details_file=None,
                  gt=None,
                  pred_bbox=None,
                  pred_mask=None,
                  iou_thresh=0.5,
                  save_dir='./'):
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
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()

    def _summarize(coco_gt, ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = coco_gt.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = coco_gt.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
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

    def cal_pr(coco_gt, coco_dt, iou_thresh, save_dir, style='bbox'):
        from pycocotools.cocoeval import COCOeval
        coco_dt = loadRes(coco_gt, coco_dt)
        np.linspace = fixed_linspace
        coco_eval = COCOeval(coco_gt, coco_dt, style)
        coco_eval.params.iouThrs = np.linspace(
            iou_thresh, iou_thresh, 1, endpoint=True)
        np.linspace = backup_linspace
        coco_eval.evaluate()
        coco_eval.accumulate()
        stats = _summarize(coco_eval, iouThr=iou_thresh)
        catIds = coco_gt.getCatIds()
        if len(catIds) != coco_eval.eval['precision'].shape[2]:
            raise Exception(
                "The category number must be same as the third dimension of precisions."
            )
        x = np.arange(0.0, 1.01, 0.01)
        color_map = get_color_map_list(256)[1:256]

        plt.subplot(1, 2, 1)
        plt.title(style + " precision-recall IoU={}".format(iou_thresh))
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.xlim(0, 1.01)
        plt.ylim(0, 1.01)
        plt.grid(linestyle='--', linewidth=1)
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
        my_x_ticks = np.arange(0, 1.01, 0.1)
        my_y_ticks = np.arange(0, 1.01, 0.1)
        plt.xticks(my_x_ticks, fontsize=5)
        plt.yticks(my_y_ticks, fontsize=5)
        for idx, catId in enumerate(catIds):
            pr_array = coco_eval.eval['precision'][0, :, idx, 0, 2]
            precision = pr_array[pr_array > -1]
            ap = np.mean(precision) if precision.size else float('nan')
            nm = coco_gt.loadCats(catId)[0]['name'] + ' AP={:0.2f}'.format(
                float(ap * 100))
            color = tuple(color_map[idx])
            color = [float(c) / 255 for c in color]
            color.append(0.75)
            plt.plot(x, pr_array, color=color, label=nm, linewidth=1)
        plt.legend(loc="lower left", fontsize=5)

        plt.subplot(1, 2, 2)
        plt.title(style + " score-recall IoU={}".format(iou_thresh))
        plt.xlabel('recall')
        plt.ylabel('score')
        plt.xlim(0, 1.01)
        plt.ylim(0, 1.01)
        plt.grid(linestyle='--', linewidth=1)
        plt.xticks(my_x_ticks, fontsize=5)
        plt.yticks(my_y_ticks, fontsize=5)
        for idx, catId in enumerate(catIds):
            nm = coco_gt.loadCats(catId)[0]['name']
            sr_array = coco_eval.eval['scores'][0, :, idx, 0, 2]
            color = tuple(color_map[idx])
            color = [float(c) / 255 for c in color]
            color.append(0.75)
            plt.plot(x, sr_array, color=color, label=nm, linewidth=1)
        plt.legend(loc="lower right", fontsize=5)
        plt.savefig(
            os.path.join(save_dir, "./{}_pr_curve(iou-{}).png".format(
                style, iou_thresh)),
            dpi=800)
        plt.close()

    cal_pr(coco, pred_bbox, iou_thresh, save_dir, style='bbox')
    if pred_mask is not None:
        cal_pr(coco, pred_mask, iou_thresh, save_dir, style='segm')
