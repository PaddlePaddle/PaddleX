# coding:utf-8

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import shutil
import paddlex as pdx


def cal_image_level(model, dataset_dir):
    file_list = os.path.join(dataset_dir, 'test_list_2.txt')
    threshold = 0.4
    # threshold = 0.5
    matrix = [[0, 0], [0, 0]]
    fire_to_no = []
    no_to_fire = []

    # 观察结果错误的图片
    fire_to_no_path = 'metric_fire_to_no'
    no_to_fire_path = 'metric_no_to_fire'
    if not os.path.exists(fire_to_no_path):
        os.makedirs(fire_to_no_path)
    else:
        shutil.rmtree(fire_to_no_path)
        os.makedirs(fire_to_no_path)

    if not os.path.exists(no_to_fire_path):
        os.makedirs(no_to_fire_path)
    else:
        shutil.rmtree(no_to_fire_path)
        os.makedirs(no_to_fire_path)

    with open(file_list, 'r') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            img_file, label = line.strip().split()[:2]
            img_file = os.path.join(dataset_dir, img_file)
            label = int(label)
            res = model.predict(img_file)
            keep_results = []
            areas = []
            for dt in res:
                cname, bbox, score = dt['category'], dt['bbox'], dt['score']
                if score < threshold:
                    continue
                keep_results.append(dt)
                areas.append(bbox[2] * bbox[3])
            areas = np.asarray(areas)
            sorted_idxs = np.argsort(-areas).tolist()
            keep_results = [keep_results[k]
                            for k in sorted_idxs] if keep_results else []

            if len(keep_results) > 0:
                predict_label = 1
            else:
                predict_label = 0

            if label == 1:
                if label == predict_label:
                    matrix[0][0] += 1
                else:
                    matrix[1][0] += 1
                    fire_to_no.append(img_file)
                    name = os.path.basename(img_file)
                    shutil.copyfile(img_file,
                                    os.path.join(fire_to_no_path, name))
            else:
                if label == predict_label:
                    matrix[1][1] += 1
                else:
                    matrix[0][1] += 1
                    no_to_fire.append(img_file)
                    # 绘制结果
                    pdx.det.visualize(
                        img_file,
                        keep_results,
                        threshold=threshold,
                        save_dir=no_to_fire_path)

    recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])
    error = matrix[0][1] / (matrix[0][1] + matrix[1][1])
    print('===matrix:', matrix)
    print('===recall:', recall)
    print('===error:', error)
    print('===烟火图被判定为无烟火的图片包含：', len(fire_to_no))
    print('===无烟火图被判定为烟火的图片包含：', len(no_to_fire))
    return recall, error


def select_best(dataset_dir, model_dirs):
    max_recall = 0
    min_error = 100
    best_recall = [0, 0]
    best_error = [0, 0]
    for model_dir in sorted(os.listdir(model_dirs)):
        if 'epoch' in model_dir or 'best_model' in model_dir:
            model_dir = os.path.join(model_dirs, model_dir)
            model = pdx.load_model(model_dir)
            recall, error = cal_image_level(model, dataset_dir)
            if recall > max_recall:
                best_recall = [model_dir, recall, error] + best_recall
                max_recall = recall
            if error < min_error:
                best_error = [model_dir, recall, error] + best_error
                min_error = error
        else:
            continue
    print('==best recall:', best_recall[:-2])
    print('====best error:', best_error[:-2])
    print('====final best:', best_recall[0], best_error[0])


if __name__ == '__main__':
    dataset_dir = 'eval_imgs'
    model_dirs = 'output/ppyolov2_r50vd_dcn/'
    select_best(dataset_dir, model_dirs)

    # # model_dir = 'output/ppyolov2_r50vd_dcn/best_model/'
    # model = pdx.load_model(model_dir)
    # recall, error = cal_image_level(model, dataset_dir)
