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
from PIL import Image, ImageDraw


def visualize_detection(image, result, threshold=0.5, save_dir=None):
    """
        Visualize bbox and mask results
    """

    image_name = os.path.split(image)[-1]
    image = Image.open(image).convert('RGB')
    image = draw_bbox_mask(image, results, threshold=threshold)
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
