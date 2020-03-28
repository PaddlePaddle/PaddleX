#!/usr/bin/env python
# coding: utf-8
import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.ImageDraw


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)



def images(data, num):
    image = {}
    image['height'] = data['imageHeight']
    image['width'] = data['imageWidth']
    image['id'] = num + 1
    image['file_name'] = data['imagePath'].split('/')[-1]
    return image


def categories(label, labels_list):
    category = {}
    category['supercategory'] = 'component'
    category['id'] = len(labels_list) + 1
    category['name'] = label
    return category


def annotations_rectangle(iscrowd, points, label, num, label_to_num, count):
    annotation = {}
    seg_points = np.asarray(points).copy()
    seg_points[1, :] = np.asarray(points)[2, :]
    seg_points[2, :] = np.asarray(points)[1, :]
    annotation['segmentation'] = [list(seg_points.flatten())]
    annotation['iscrowd'] = iscrowd
    annotation['image_id'] = num + 1
    annotation['bbox'] = list(
        map(
            float,
            [
                points[0][0],
                points[0][1],
                points[1][0] - points[0][0],
                points[1][1] - points[0][1],
            ], ), )
    annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
    annotation['category_id'] = label_to_num[label]
    annotation['id'] = count
    return annotation


def annotations_polygon(annotation, iscrowd, height, width, points, label, num,
                        label_to_num, count):
    
    if len(annotation) == 0:
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = iscrowd
        annotation['image_id'] = num + 1
        annotation['bbox'] = list(map(float, get_bbox(height, width, points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        annotation['category_id'] = label_to_num[label]
        annotation['id'] = count
    else:
        annotation['segmentation'].append(list(np.asarray(points).flatten()))
        box1 = annotation['bbox']
        box2 = list(map(float, get_bbox(height, width, points)))
        x11, y11, x12, y12 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
        x21, y21, x22, y22 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
        x1 = x21 if x11 > x21 else x11
        y1 = y21 if y11 > y21 else y11
        x2 = x22 if x12 < x22 else x12
        y2 = y22 if y12 < y22 else y12
        annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]



def get_bbox(height, width, points):
    polygons = points
    mask = np.zeros([height, width], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    left_top_r = np.min(rows)
    left_top_c = np.min(clos)
    right_bottom_r = np.max(rows)
    right_bottom_c = np.max(clos)
    return [
        left_top_c,
        left_top_r,
        right_bottom_c - left_top_c,
        right_bottom_r - left_top_r,
    ]


def deal_json(img_path, json_path):
    data_coco = {}
    label_to_num = {}
    images_list = []
    categories_list = []
    annotations_list = []
    labels_list = []
    num = -1
    for img_file in os.listdir(img_path):
        img_label = img_file.split('.')[0]
        if img_label == '':
            continue
        label_file = osp.join(json_path, img_label + '.json')
        assert os.path.exists(label_file), \
            'The .json file of {} is not exists!'.format(img_file)
        print('Generating dataset from:', label_file)
        num = num + 1
        with open(label_file) as f:
            data = json.load(f)
            images_list.append(images(data, num))
            count = 0
            lmid_count = {}
            for shapes in data['shapes']:
                count += 1
                label = shapes['label']
                part = label.split('_')
                iscrowd = int(part[-1][0])
                label = label.split('_' + part[-1])[0]
                if label not in labels_list:
                    categories_list.append(categories(label, labels_list))
                    labels_list.append(label)
                    label_to_num[label] = len(labels_list)
                points = shapes['points']
                p_type = shapes['shape_type']
                if p_type == 'polygon':
                    if len(part[-1]) > 1:
                        lmid = part[-1][1:]
                        if lmid in lmid_count:
                            real_count = lmid_count[lmid]
                            real_anno = None
                            for anno in annotations_list:
                                if anno['id'] == real_count:
                                    real_anno = anno
                                    break
                            annotations_polygon(anno, iscrowd, data['imageHeight'], data[
                                'imageWidth'], points, label, num, label_to_num,
                                                real_count)
                            count -= 1
                        else:
                            lmid_count[lmid] = count
                            anno = {}
                            annotations_polygon(anno, iscrowd, data['imageHeight'], data[
                                'imageWidth'], points, label, num, label_to_num,
                                                count)
                            annotations_list.append(anno)
                    else:
                        anno = {}
                        annotations_polygon(anno, iscrowd, data['imageHeight'], data[
                            'imageWidth'], points, label, num, label_to_num,
                                            count)
                        annotations_list.append(anno)
                if p_type == 'rectangle':
                    points.append([points[0][0], points[1][1]])
                    points.append([points[1][0], points[0][1]])
                    annotations_list.append(
                        annotations_rectangle(iscrowd, points, label, num,
                                              label_to_num, count))
    data_coco['images'] = images_list
    data_coco['categories'] = categories_list
    data_coco['annotations'] = annotations_list
    return data_coco


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument('--json_input_dir', help='input annotated directory')
    parser.add_argument('--image_input_dir', help='image directory')
    args = parser.parse_args()
    try:
        assert os.path.exists(args.json_input_dir)
    except AssertionError as e:
        print('The json folder does not exist!')
        os._exit(0)
    try:
        assert os.path.exists(args.image_input_dir)
    except AssertionError as e:
        print('The image folder does not exist!')
        os._exit(0)

    # Allocate the dataset.
    total_num = len(glob.glob(osp.join(args.json_input_dir, '*.json')))

    # Deal with the json files.
    res_dir = os.path.abspath(os.path.join(args.image_input_dir, '..'))
    if not os.path.exists(res_dir + '/annotations'):
        os.makedirs(res_dir + '/annotations')
    train_data_coco = deal_json(args.image_input_dir, args.json_input_dir)
    train_json_path = osp.join(
        res_dir + '/annotations',
        'instance_{}.json'.format(
            os.path.basename(os.path.abspath(args.image_input_dir))))
    json.dump(
        train_data_coco,
        open(
            train_json_path,
            'w'),
        indent=4,
        cls=MyEncoder)


if __name__ == '__main__':
    main()
