# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import cv2
import numpy as np
import time
import pycocotools.mask as mask_util
import paddlex.utils.logging as logging
from paddlex.utils import is_pic
from .det_metrics.coco_utils import loadRes


def visualize_detection(image,
                        result,
                        threshold=0.5,
                        save_dir='./',
                        color=None):
    """
        Visualize bbox and mask results
    """

    if isinstance(image, np.ndarray):
        image_name = str(int(time.time() * 1000)) + '.jpg'
    else:
        image_name = os.path.split(image)[-1]
        image = cv2.imread(image)
    image = draw_bbox_mask(image, result, threshold=threshold, color_map=color)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        out_path = os.path.join(save_dir, 'visualize_{}'.format(image_name))
        cv2.imwrite(out_path, image)
        logging.info('The visualized result is saved at {}'.format(out_path))
    else:
        return image


def visualize_segmentation(image,
                           result,
                           weight=0.6,
                           save_dir='./',
                           color=None):
    """
    Convert segment result to color image, and save added image.
    Args:
        image: the path of origin image
        result: the predict result of image
        weight: the image weight of visual image, and the result weight is (1 - weight)
        save_dir: the directory for saving visual image
        color: the list of a BGR-mode color for each label.
    """
    label_map = result['label_map']
    color_map = get_color_map_list(256)
    if color is not None:
        for i in range(len(color) // 3):
            color_map[i] = color[i * 3:(i + 1) * 3]
    color_map = np.array(color_map).astype("uint8")

    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(label_map, color_map[:, 0])
    c2 = cv2.LUT(label_map, color_map[:, 1])
    c3 = cv2.LUT(label_map, color_map[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))

    if isinstance(image, np.ndarray):
        im = image
        image_name = str(int(time.time() * 1000)) + '.jpg'
        if image.shape[2] != 3:
            logging.info(
                "The image is not 3-channel array, so predicted label map is shown as a pseudo color image."
            )
            weight = 0.
    else:
        image_name = os.path.split(image)[-1]
        if not is_pic(image):
            logging.info(
                "The image cannot be opened by opencv, so predicted label map is shown as a pseudo color image."
            )
            image_name = image_name.split('.')[0] + '.jpg'
            weight = 0.
        else:
            im = cv2.imread(image)

    if abs(weight) < 1e-5:
        vis_result = pseudo_img
    else:
        vis_result = cv2.addWeighted(im, weight,
                                     pseudo_img.astype(im.dtype), 1 - weight,
                                     0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        out_path = os.path.join(save_dir, 'visualize_{}'.format(image_name))
        cv2.imwrite(out_path, vis_result)
        logging.info('The visualized result is saved as {}'.format(out_path))
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


def draw_bbox_mask(image, results, threshold=0.5, color_map=None):
    _SMALL_OBJECT_AREA_THRESH = 1000
    height, width = image.shape[:2]
    default_font_scale = max(np.sqrt(height * width) // 900, .5)
    linewidth = max(default_font_scale / 40, 2)

    labels = list()
    for dt in results:
        if dt['category'] not in labels:
            labels.append(dt['category'])

    if color_map is None:
        color_map = get_color_map_list(len(labels) + 2)[2:]
    else:
        color_map = np.asarray(color_map)
        if color_map.shape[0] != len(labels) or color_map.shape[1] != 3:
            raise Exception(
                "The shape for color_map is required to be {}x3, but recieved shape is {}x{}.".
                format(len(labels), color_map.shape))
        if np.max(color_map) > 255 or np.min(color_map) < 0:
            raise ValueError(
                " The values in color_map should be within 0-255 range.")

    keep_results = []
    areas = []
    for dt in results:
        cname, bbox, score = dt['category'], dt['bbox'], dt['score']
        if score < threshold:
            continue
        keep_results.append(dt)
        areas.append(bbox[2] * bbox[3])
    areas = np.asarray(areas)
    sorted_idxs = np.argsort(-areas).tolist()
    keep_results = [keep_results[k]
                    for k in sorted_idxs] if keep_results else []

    for dt in keep_results:
        cname, bbox, score = dt['category'], dt['bbox'], dt['score']
        bbox = list(map(int, bbox))
        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h

        color = tuple(color_map[labels.index(cname)])
        # draw bbox
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color,
                              linewidth)

        # draw mask
        if 'mask' in dt:
            mask = mask_util.decode(dt['mask']) * 255
            image = image.astype('float32')
            alpha = .7
            w_ratio = .4
            color_mask = np.asarray(color, dtype=np.int)
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
            idx = np.nonzero(mask)
            image[idx[0], idx[1], :] *= 1.0 - alpha
            image[idx[0], idx[1], :] += alpha * color_mask
            image = image.astype("uint8")
            contours = cv2.findContours(
                mask.astype("uint8"), cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_NONE)[-2]
            image = cv2.drawContours(
                image,
                contours,
                contourIdx=-1,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA)

        # draw label
        text_pos = (xmin, ymin)
        instance_area = w * h
        if (instance_area < _SMALL_OBJECT_AREA_THRESH or h < 40):
            if ymin >= height - 5:
                text_pos = (xmin, ymin)
            else:
                text_pos = (xmin, ymax)
        height_ratio = h / np.sqrt(height * width)
        font_scale = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2,
                              2) * 0.5 * default_font_scale)
        text = "{} {:.2f}".format(cname, score)
        (tw, th), baseline = cv2.getTextSize(
            text,
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=font_scale,
            thickness=1)
        image = cv2.rectangle(
            image,
            text_pos, (text_pos[0] + tw, text_pos[1] + th + baseline),
            color=color,
            thickness=-1)
        image = cv2.putText(
            image,
            text, (text_pos[0], text_pos[1] + th),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA)

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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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
        coco_dt = loadRes(coco_gt, coco_dt)
        coco_eval = COCOeval(coco_gt, coco_dt, style)
        coco_eval.params.iouThrs = np.linspace(
            iou_thresh, iou_thresh, 1, endpoint=True)
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
        plt.legend(loc="lower left", fontsize=5)
        plt.savefig(
            os.path.join(
                save_dir,
                "./{}_pr_curve(iou-{}).png".format(style, iou_thresh)),
            dpi=800)
        plt.close()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cal_pr(coco, pred_bbox, iou_thresh, save_dir, style='bbox')
    if pred_mask is not None:
        cal_pr(coco, pred_mask, iou_thresh, save_dir, style='segm')
