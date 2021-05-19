# copytrue (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os.path as osp
import os
import numpy as np
from PIL import Image
import sys
import cv2
import psutil
import shutil
import pickle
import base64
import multiprocessing as mp
from ..utils import (pkill, set_folder_status, get_folder_status, TaskStatus,
                     PredictStatus, PruneStatus)
from .evaluate.draw_pred_result import visualize_classified_result, visualize_detected_result, visualize_segmented_result
from .visualize import plot_det_label, plot_insseg_label, get_color_map_list


def _call_paddle_prune(best_model_path, prune_analysis_path, params):
    mode = 'w'
    sys.stdout = open(
        osp.join(prune_analysis_path, 'out.log'), mode, encoding='utf-8')
    sys.stderr = open(
        osp.join(prune_analysis_path, 'err.log'), mode, encoding='utf-8')
    sensitivities_path = osp.join(prune_analysis_path, "sensitivities.data")
    task_type = params['task_type']
    dataset_path = params['dataset_path']
    os.environ['CUDA_VISIBLE_DEVICES'] = params['train'].cuda_visible_devices
    if task_type == "classification":
        from .prune.classification import prune
    elif task_type in ["detection", "instance_segmentation"]:
        from .prune.detection import prune
    elif task_type == "segmentation":
        from .prune.segmentation import prune
    batch_size = params['train'].batch_size
    prune(best_model_path, dataset_path, sensitivities_path, batch_size)
    import paddlex as pdx
    from paddlex.cv.models.slim.visualize import visualize
    model = pdx.load_model(best_model_path)
    visualize(model, sensitivities_path, prune_analysis_path)
    set_folder_status(prune_analysis_path, PruneStatus.XSPRUNEDONE)


def _call_paddlex_train(task_path, params):
    '''
    Args:
        params为dict，字段包括'pretrain_weights_download_save_dir': 预训练模型保存路径，
        'task_type': 任务类型，'dataset_path': 数据集路径，'train':训练参数
    '''

    mode = 'w'
    if params['train'].resume_checkpoint is not None:
        mode = 'a'
    sys.stdout = open(osp.join(task_path, 'out.log'), mode, encoding='utf-8')
    sys.stderr = open(osp.join(task_path, 'err.log'), mode, encoding='utf-8')
    sys.stdout.write("This log file path is {}\n".format(
        osp.join(task_path, 'out.log')))
    sys.stdout.write("注意：标志为WARNING/INFO类的仅为警告或提示类信息，非错误信息\n")
    sys.stderr.write("This log file path is {}\n".format(
        osp.join(task_path, 'err.log')))
    sys.stderr.write("注意：标志为WARNING/INFO类的仅为警告或提示类信息，非错误信息\n")
    os.environ['CUDA_VISIBLE_DEVICES'] = params['train'].cuda_visible_devices
    import paddlex as pdx
    pdx.gui_mode = True
    pdx.log_level = 3
    pdx.pretrain_dir = params['pretrain_weights_download_save_dir']
    task_type = params['task_type']
    dataset_path = params['dataset_path']
    if task_type == "classification":
        from .train.classification import train
    elif task_type in ["detection", "instance_segmentation"]:
        from .train.detection import train
    elif task_type == "segmentation":
        from .train.segmentation import train
    train(task_path, dataset_path, params['train'])
    set_folder_status(task_path, TaskStatus.XTRAINDONE)


def _call_paddlex_evaluate_model(task_path,
                                 model_path,
                                 task_type,
                                 epoch,
                                 topk=5,
                                 score_thresh=0.3,
                                 overlap_thresh=0.5):
    evaluate_status_path = osp.join(task_path, './logs/evaluate')
    sys.stdout = open(
        osp.join(evaluate_status_path, 'out.log'), 'w', encoding='utf-8')
    sys.stderr = open(
        osp.join(evaluate_status_path, 'err.log'), 'w', encoding='utf-8')
    if task_type == "classification":
        from .evaluate.classification import Evaluator
        evaluator = Evaluator(model_path, topk=topk)
    elif task_type == "detection":
        from .evaluate.detection import DetEvaluator
        evaluator = DetEvaluator(
            model_path,
            score_threshold=score_thresh,
            overlap_thresh=overlap_thresh)
    elif task_type == "instance_segmentation":
        from .evaluate.detection import InsSegEvaluator
        evaluator = InsSegEvaluator(
            model_path,
            score_threshold=score_thresh,
            overlap_thresh=overlap_thresh)
    elif task_type == "segmentation":
        from .evaluate.segmentation import Evaluator
        evaluator = Evaluator(model_path)
    report = evaluator.generate_report()
    report['epoch'] = epoch
    pickle.dump(report, open(osp.join(task_path, "eval_res.pkl"), "wb"))
    set_folder_status(evaluate_status_path, TaskStatus.XEVALUATED)
    set_folder_status(task_path, TaskStatus.XEVALUATED)


def _call_paddlex_predict(task_path,
                          predict_status_path,
                          params,
                          img_list,
                          img_data,
                          save_dir,
                          score_thresh,
                          epoch=None):
    total_num = open(
        osp.join(predict_status_path, 'total_num'), 'w', encoding='utf-8')

    def write_file_num(total_file_num):
        total_num.write(str(total_file_num))
        total_num.close()

    sys.stdout = open(
        osp.join(predict_status_path, 'out.log'), 'w', encoding='utf-8')
    sys.stderr = open(
        osp.join(predict_status_path, 'err.log'), 'w', encoding='utf-8')

    import paddlex as pdx
    pdx.log_level = 3
    task_type = params['task_type']
    dataset_path = params['dataset_path']
    if epoch is None:
        model_path = osp.join(task_path, 'output', 'best_model')
    else:
        model_path = osp.join(task_path, 'output', 'epoch_{}'.format(epoch))
    model = pdx.load_model(model_path)
    file_list = dict()
    predicted_num = 0
    if task_type == "classification":
        if img_data is None:
            if len(img_list) == 0 and osp.exists(
                    osp.join(dataset_path, "test_list.txt")):
                with open(osp.join(dataset_path, "test_list.txt")) as f:
                    for line in f:
                        items = line.strip().split()
                        file_list[osp.join(dataset_path, items[0])] = items[1]
            else:
                for image in img_list:
                    file_list[image] = None
            total_file_num = len(file_list)
            write_file_num(total_file_num)
            for image, label_id in file_list.items():
                pred_result = {}
                if label_id is not None:
                    pred_result["gt_label"] = model.labels[int(label_id)]
                results = model.predict(img_file=image)
                pred_result["label"] = []
                pred_result["score"] = []
                pred_result["topk"] = len(results)
                for res in results:
                    pred_result["label"].append(res['category'])
                    pred_result["score"].append(res['score'])
                visualize_classified_result(save_dir, image, pred_result)
                predicted_num += 1
        else:
            img_data = base64.b64decode(img_data)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
            results = model.predict(img)
            pred_result = {}
            pred_result["label"] = []
            pred_result["score"] = []
            pred_result["topk"] = len(results)
            for res in results:
                pred_result["label"].append(res['category'])
                pred_result["score"].append(res['score'])
            visualize_classified_result(save_dir, img, pred_result)
    elif task_type in ["detection", "instance_segmentation"]:
        if img_data is None:
            if task_type == "detection" and osp.exists(
                    osp.join(dataset_path, "test_list.txt")):
                if len(img_list) == 0 and osp.exists(
                        osp.join(dataset_path, "test_list.txt")):
                    with open(osp.join(dataset_path, "test_list.txt")) as f:
                        for line in f:
                            items = line.strip().split()
                            file_list[osp.join(dataset_path, items[0])] = \
                                osp.join(dataset_path, items[1])
                else:
                    for image in img_list:
                        file_list[image] = None
                total_file_num = len(file_list)
                write_file_num(total_file_num)
                for image, anno in file_list.items():
                    results = model.predict(img_file=image)
                    image_pred = pdx.det.visualize(
                        image, results, threshold=score_thresh, save_dir=None)
                    save_name = osp.join(save_dir, osp.split(image)[-1])
                    image_gt = None
                    if anno is not None:
                        image_gt = plot_det_label(image, anno, model.labels)
                    visualize_detected_result(save_name, image_gt, image_pred)
                    predicted_num += 1
            elif len(img_list) == 0 and osp.exists(
                    osp.join(dataset_path, "test.json")):
                from pycocotools.coco import COCO
                anno_path = osp.join(dataset_path, "test.json")
                coco = COCO(anno_path)
                img_ids = coco.getImgIds()
                total_file_num = len(img_ids)
                write_file_num(total_file_num)
                for img_id in img_ids:
                    img_anno = coco.loadImgs(img_id)[0]
                    file_name = img_anno['file_name']
                    name = (osp.split(file_name)[-1]).split(".")[0]
                    anno = osp.join(dataset_path, "Annotations", name + ".npy")
                    img_file = osp.join(dataset_path, "JPEGImages", file_name)
                    results = model.predict(img_file=img_file)
                    image_pred = pdx.det.visualize(
                        img_file,
                        results,
                        threshold=score_thresh,
                        save_dir=None)
                    save_name = osp.join(save_dir, osp.split(img_file)[-1])
                    if task_type == "detection":
                        image_gt = plot_det_label(img_file, anno, model.labels)
                    else:
                        image_gt = plot_insseg_label(img_file, anno,
                                                     model.labels)
                    visualize_detected_result(save_name, image_gt, image_pred)
                    predicted_num += 1
            else:
                total_file_num = len(img_list)
                write_file_num(total_file_num)
                for image in img_list:
                    results = model.predict(img_file=image)
                    image_pred = pdx.det.visualize(
                        image, results, threshold=score_thresh, save_dir=None)
                    save_name = osp.join(save_dir, osp.split(image)[-1])
                    visualize_detected_result(save_name, None, image_pred)
                    predicted_num += 1
        else:
            img_data = base64.b64decode(img_data)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
            results = model.predict(img)
            image_pred = pdx.det.visualize(
                img, results, threshold=score_thresh, save_dir=None)
            image_gt = None
            save_name = osp.join(save_dir, 'predict_result.png')
            visualize_detected_result(save_name, image_gt, image_pred)

    elif task_type == "segmentation":
        if img_data is None:
            if len(img_list) == 0 and osp.exists(
                    osp.join(dataset_path, "test_list.txt")):
                with open(osp.join(dataset_path, "test_list.txt")) as f:
                    for line in f:
                        items = line.strip().split()
                        file_list[osp.join(dataset_path, items[0])] = \
                            osp.join(dataset_path, items[1])
            else:
                for image in img_list:
                    file_list[image] = None
            total_file_num = len(file_list)
            write_file_num(total_file_num)
            color_map = get_color_map_list(256)
            legend = {}
            for i in range(len(model.labels)):
                legend[model.labels[i]] = color_map[i]
            for image, anno in file_list.items():
                results = model.predict(img_file=image)
                image_pred = pdx.seg.visualize(image, results, save_dir=None)
                pse_pred = pdx.seg.visualize(
                    image, results, weight=0, save_dir=None)
                image_ground = None
                pse_label = None
                if anno is not None:
                    label = np.asarray(Image.open(anno)).astype('uint8')
                    image_ground = pdx.seg.visualize(
                        image, {'label_map': label}, save_dir=None)
                    pse_label = pdx.seg.visualize(
                        image, {'label_map': label}, weight=0, save_dir=None)
                save_name = osp.join(save_dir, osp.split(image)[-1])
                visualize_segmented_result(save_name, image_ground, pse_label,
                                           image_pred, pse_pred, legend)
                predicted_num += 1
        else:
            img_data = base64.b64decode(img_data)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
            color_map = get_color_map_list(256)
            legend = {}
            for i in range(len(model.labels)):
                legend[model.labels[i]] = color_map[i]
            results = model.predict(img)
            image_pred = pdx.seg.visualize(image, results, save_dir=None)
            pse_pred = pdx.seg.visualize(
                image, results, weight=0, save_dir=None)
            image_ground = None
            pse_label = None
            save_name = osp.join(save_dir, 'predict_result.png')
            visualize_segmented_result(save_name, image_ground, pse_label,
                                       image_pred, pse_pred, legend)
    set_folder_status(predict_status_path, PredictStatus.XPREDONE)


def _call_paddlex_export_infer(task_path, save_dir, export_status_path, epoch):
    # 导出模型不使用GPU
    sys.stdout = open(
        osp.join(export_status_path, 'out.log'), 'w', encoding='utf-8')
    sys.stderr = open(
        osp.join(export_status_path, 'err.log'), 'w', encoding='utf-8')
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import paddlex as pdx
    if epoch is not None:
        model_dir = "epoch_{}".format(epoch)
        model_path = osp.join(task_path, 'output', model_dir)
    else:
        model_path = osp.join(task_path, 'output', 'best_model')
    model = pdx.load_model(model_path)
    model.export_inference_model(save_dir)
    set_folder_status(export_status_path, TaskStatus.XEXPORTED)
    set_folder_status(task_path, TaskStatus.XEXPORTED)


def _call_paddlex_export_quant(task_path, params, save_dir, export_status_path,
                               epoch):
    sys.stdout = open(
        osp.join(export_status_path, 'out.log'), 'w', encoding='utf-8')
    sys.stderr = open(
        osp.join(export_status_path, 'err.log'), 'w', encoding='utf-8')
    dataset_path = params['dataset_path']
    task_type = params['task_type']
    os.environ['CUDA_VISIBLE_DEVICES'] = params['train'].cuda_visible_devices
    import paddlex as pdx
    if epoch is not None:
        model_dir = "epoch_{}".format(epoch)
        model_path = osp.join(task_path, 'output', model_dir)
    else:
        model_path = osp.join(task_path, 'output', 'best_model')
    model = pdx.load_model(model_path)
    if task_type == "classification":
        train_file_list = osp.join(dataset_path, 'train_list.txt')
        val_file_list = osp.join(dataset_path, 'val_list.txt')
        label_list = osp.join(dataset_path, 'labels.txt')
        quant_dataset = pdx.datasets.ImageNet(
            data_dir=dataset_path,
            file_list=train_file_list,
            label_list=label_list,
            transforms=model.test_transforms)
        eval_dataset = pdx.datasets.ImageNet(
            data_dir=dataset_path,
            file_list=val_file_list,
            label_list=label_list,
            transforms=model.eval_transforms)
    elif task_type == "detection":
        train_file_list = osp.join(dataset_path, 'train_list.txt')
        val_file_list = osp.join(dataset_path, 'val_list.txt')
        label_list = osp.join(dataset_path, 'labels.txt')
        quant_dataset = pdx.datasets.VOCDetection(
            data_dir=dataset_path,
            file_list=train_file_list,
            label_list=label_list,
            transforms=model.test_transforms)
        eval_dataset = pdx.datasets.VOCDetection(
            data_dir=dataset_path,
            file_list=val_file_list,
            label_list=label_list,
            transforms=model.eval_transforms)
    elif task_type == "instance_segmentation":
        train_json = osp.join(dataset_path, 'train.json')
        val_json = osp.join(dataset_path, 'val.json')
        quant_dataset = pdx.datasets.CocoDetection(
            data_dir=osp.join(dataset_path, 'JPEGImages'),
            ann_file=train_json,
            transforms=model.test_transforms)
        eval_dataset = pdx.datasets.CocoDetection(
            data_dir=osp.join(dataset_path, 'JPEGImages'),
            ann_file=val_json,
            transforms=model.eval_transforms)
    elif task_type == "segmentation":
        train_file_list = osp.join(dataset_path, 'train_list.txt')
        val_file_list = osp.join(dataset_path, 'val_list.txt')
        label_list = osp.join(dataset_path, 'labels.txt')
        quant_dataset = pdx.datasets.SegDataset(
            data_dir=dataset_path,
            file_list=train_file_list,
            label_list=label_list,
            transforms=model.test_transforms)
        eval_dataset = pdx.datasets.SegDataset(
            data_dir=dataset_path,
            file_list=val_file_list,
            label_list=label_list,
            transforms=model.eval_transforms)
    metric_before = model.evaluate(eval_dataset)
    pdx.log_level = 3
    pdx.slim.export_quant_model(
        model, quant_dataset, batch_size=1, save_dir=save_dir, cache_dir=None)
    model_quant = pdx.load_model(save_dir)
    metric_after = model_quant.evaluate(eval_dataset)
    metrics = {}
    if task_type == "segmentation":
        metrics['before'] = {'miou': metric_before['miou']}
        metrics['after'] = {'miou': metric_after['miou']}
    else:
        metrics['before'] = metric_before
        metrics['after'] = metric_after
    import json
    with open(
            osp.join(export_status_path, 'quant_result.json'),
            'w',
            encoding='utf-8') as f:
        json.dump(metrics, f)
    set_folder_status(export_status_path, TaskStatus.XEXPORTED)
    set_folder_status(task_path, TaskStatus.XEXPORTED)


def _call_paddlelite_export_lite(model_path, save_dir=None, place="arm"):
    import paddlelite.lite as lite
    opt = lite.Opt()
    model_file = os.path.join(model_path, '__model__')
    params_file = os.path.join(model_path, '__params__')
    if save_dir is None:
        save_dir = osp.join(model_path, "lite_model")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    path = osp.join(save_dir, "model")
    opt.run_optimize("", model_file, params_file, "naive_buffer", place, path)


def safe_clean_folder(folder):
    if osp.exists(folder):
        try:
            shutil.rmtree(folder)
            os.makedirs(folder)
        except Exception as e:
            pass
        if osp.exists(folder):
            for root, dirs, files in os.walk(folder):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except Exception as e:
                        pass
        else:
            os.makedirs(folder)
    else:
        os.makedirs(folder)
    if not osp.exists(folder):
        os.makedirs(folder)


def get_task_max_saved_epochs(task_path):
    saved_epoch_num = -1
    output_path = osp.join(task_path, "output")
    if osp.exists(output_path):
        for f in os.listdir(output_path):
            if f.startswith("epoch_"):
                if not osp.exists(osp.join(output_path, f, '.success')):
                    continue
                curr_epoch_num = int(f[6:])
                if curr_epoch_num > saved_epoch_num:
                    saved_epoch_num = curr_epoch_num
    return saved_epoch_num


def get_task_status(task_path):
    status, message = get_folder_status(task_path, True)
    task_id = os.path.split(task_path)[-1]
    err_log = os.path.join(task_path, 'err.log')
    if status in [TaskStatus.XTRAINING, TaskStatus.XPRUNETRAIN]:
        pid = int(message)
        is_dead = False
        if not psutil.pid_exists(pid):
            is_dead = True
        else:
            p = psutil.Process(pid)
            if p.status() == 'zombie':
                is_dead = True
        if is_dead:
            status = TaskStatus.XTRAINFAIL
            message = "训练任务{}异常终止，请查阅错误日志具体确认原因{}。\n\n 如若通过日志无法确定原因，可尝试以下几种方法，\n" \
            "1. 尝试重新启动训练，看是否能正常训练; \n" \
            "2. 调低batch_size（需同时按比例调低学习率等参数）排除是否是显存或内存不足的原因导致;\n" \
            "3. 前往GitHub提ISSUE，描述清楚问题会有工程师及时回复： https://github.com/PaddlePaddle/PaddleX/issues ; \n" \
            "3. 加QQ群1045148026或邮件至paddlex@baidu.com在线咨询工程师".format(task_id, err_log)
            set_folder_status(task_path, status, message)
    return status, message


def train_model(task_path):
    """训练模型

    Args:
        task_path(str): 模型训练的参数保存在task_path下的'params.pkl'文件中
    """
    params_conf_file = osp.join(task_path, 'params.pkl')
    assert osp.exists(
        params_conf_file), "任务无法启动，路径{}下不存在参数配置文件params.pkl".format(task_path)
    with open(params_conf_file, 'rb') as f:
        params = pickle.load(f)
    sensitivities_path = params['train'].sensitivities_path
    p = mp.Process(target=_call_paddlex_train, args=(task_path, params))
    p.start()
    if sensitivities_path is None:
        set_folder_status(task_path, TaskStatus.XTRAINING, p.pid)
    else:
        set_folder_status(task_path, TaskStatus.XPRUNETRAIN, p.pid)
    return p


def stop_train_model(task_path):
    """停止正在训练的模型

    Args:
        task_path(str): 从task_path下的'XTRANING'文件中获取训练的进程id
    """
    status, message = get_task_status(task_path)
    if status in [TaskStatus.XTRAINING, TaskStatus.XPRUNETRAIN]:
        pid = int(message)
        pkill(pid)
        best_model_saved = True
        if not osp.exists(osp.join(task_path, 'output', 'best_model')):
            best_model_saved = False
        set_folder_status(task_path, TaskStatus.XTRAINEXIT, best_model_saved)
    else:
        raise Exception("模型训练任务没在运行中")


def prune_analysis_model(task_path):
    """模型裁剪分析

    Args:
        task_path(str): 模型训练的参数保存在task_path
        dataset_path(str) 模型裁剪中评估数据集的路径
    """
    best_model_path = osp.join(task_path, 'output', 'best_model')
    assert osp.exists(best_model_path), "该任务暂未保存模型，无法进行模型裁剪分析"
    prune_analysis_path = osp.join(task_path, 'prune')
    if not osp.exists(prune_analysis_path):
        os.makedirs(prune_analysis_path)

    params_conf_file = osp.join(task_path, 'params.pkl')
    assert osp.exists(
        params_conf_file), "任务无法启动，路径{}下不存在参数配置文件params.pkl".format(task_path)
    with open(params_conf_file, 'rb') as f:
        params = pickle.load(f)
    assert params['train'].model.lower() not in [
        "ppyolo", "fasterrcnn", "maskrcnn", "fastscnn", "HRNet_W18"
    ], "暂不支持PPYOLO、FasterRCNN、MaskRCNN、HRNet_W18、FastSCNN模型裁剪"
    p = mp.Process(
        target=_call_paddle_prune,
        args=(best_model_path, prune_analysis_path, params))
    p.start()
    set_folder_status(prune_analysis_path, PruneStatus.XSPRUNEING, p.pid)
    set_folder_status(task_path, TaskStatus.XPRUNEING, p.pid)
    return p


def get_prune_status(prune_path):
    status, message = get_folder_status(prune_path, True)
    if status in [PruneStatus.XSPRUNEING]:
        pid = int(message)
        is_dead = False
        if not psutil.pid_exists(pid):
            is_dead = True
        else:
            p = psutil.Process(pid)
            if p.status() == 'zombie':
                is_dead = True
        if is_dead:
            status = PruneStatus.XSPRUNEFAIL
            message = "模型裁剪异常终止，可能原因如下：\n1.暂不支持FasterRCNN、MaskRCNN模型的模型裁剪\n2.模型裁剪过程中进程被异常结束，建议重新启动模型裁剪任务"
            set_folder_status(prune_path, status, message)
    return status, message


def stop_prune_analysis(prune_path):
    """停止正在裁剪分析的模型

    Args:
        prune_path(str): prune_path'XSSLMING'文件中获取训练的进程id
    """
    status, message = get_prune_status(prune_path)
    if status == PruneStatus.XSPRUNEING:
        pid = int(message)
        pkill(pid)
        set_folder_status(prune_path, PruneStatus.XSPRUNEEXIT)
    else:
        raise Exception("模型裁剪分析任务未在运行中")


def evaluate_model(task_path,
                   task_type,
                   epoch=None,
                   topk=5,
                   score_thresh=0.3,
                   overlap_thresh=0.5):
    """评估最优模型

    Args:
        task_path(str): 模型训练相关结果的保存路径
    """
    output_path = osp.join(task_path, 'output')
    if not osp.exists(osp.join(output_path, 'best_model')):
        raise Exception("未在训练路径{}下发现保存的best_model，无法进行评估".format(output_path))
    evaluate_status_path = osp.join(task_path, './logs/evaluate')
    safe_clean_folder(evaluate_status_path)
    if epoch is None:
        model_path = osp.join(output_path, 'best_model')
    else:
        epoch_dir = "{}_{}".format('epoch', epoch)
        model_path = osp.join(output_path, epoch_dir)
    p = mp.Process(
        target=_call_paddlex_evaluate_model,
        args=(task_path, model_path, task_type, epoch, topk, score_thresh,
              overlap_thresh))
    p.start()
    set_folder_status(evaluate_status_path, TaskStatus.XEVALUATING, p.pid)
    return p


def get_evaluate_status(task_path):
    """获取导出状态
    Args:
        task_path(str): 训练任务文件夹
    """
    evaluate_status_path = osp.join(task_path, './logs/evaluate')
    if not osp.exists(evaluate_status_path):
        return None, "No evaluate fold in path {}".format(task_path)
    status, message = get_folder_status(evaluate_status_path, True)
    if status == TaskStatus.XEVALUATING:
        pid = int(message)
        is_dead = False
        if not psutil.pid_exists(pid):
            is_dead = True
        else:
            p = psutil.Process(pid)
            if p.status() == 'zombie':
                is_dead = True
        if is_dead:
            status = TaskStatus.XEVALUATEFAIL
            message = "评估过程出现异常，请尝试重新评估！"
            set_folder_status(evaluate_status_path, status, message)
    if status not in [
            TaskStatus.XEVALUATING, TaskStatus.XEVALUATED,
            TaskStatus.XEVALUATEFAIL
    ]:
        raise ValueError("Wrong status in evaluate task {}".format(status))
    return status, message


def get_predict_status(task_path):
    """获取预测任务状态

    Args:
        task_path(str): 从predict_path下的'XPRESTART'文件中获取训练的进程id
    """
    from ..utils import list_files
    predict_status_path = osp.join(task_path, "./logs/predict")
    save_dir = osp.join(task_path, "visualized_test_results")
    if not osp.exists(save_dir):
        return None, "任务目录下没有visualized_test_results文件夹，{}".format(
            task_path), 0, 0
    status, message = get_folder_status(predict_status_path, True)
    if status == PredictStatus.XPRESTART:
        pid = int(message)
        is_dead = False
        if not psutil.pid_exists(pid):
            is_dead = True
        else:
            p = psutil.Process(pid)
            if p.status() == 'zombie':
                is_dead = True
        if is_dead:
            status = PredictStatus.XPREFAIL
            message = "图片预测过程出现异常，请尝试重新预测！"
            set_folder_status(predict_status_path, status, message)
    if status not in [
            PredictStatus.XPRESTART, PredictStatus.XPREDONE,
            PredictStatus.XPREFAIL
    ]:
        raise ValueError("预测任务状态异常，{}".format(status))
    predict_num = len(list_files(save_dir))
    if predict_num > 0:
        if predict_num == 1:
            total_num = 1
        else:
            total_num = int(
                open(
                    osp.join(predict_status_path, "total_num"),
                    encoding='utf-8').readline().strip())
    else:
        predict_num = 0
        total_num = 0
    return status, message, predict_num, total_num


def predict_test_pics(task_path,
                      img_list=[],
                      img_data=None,
                      save_dir=None,
                      score_thresh=0.5,
                      epoch=None):
    """模型预测

    Args:
        task_path(str): 模型训练的参数保存在task_path下的'params.pkl'文件中
    """
    params_conf_file = osp.join(task_path, 'params.pkl')
    assert osp.exists(
        params_conf_file), "任务无法启动，路径{}下不存在参数配置文件params.pkl".format(task_path)
    with open(params_conf_file, 'rb') as f:
        params = pickle.load(f)
    predict_status_path = osp.join(task_path, "./logs/predict")
    safe_clean_folder(predict_status_path)
    save_dir = osp.join(task_path, 'visualized_test_results')
    safe_clean_folder(save_dir)
    p = mp.Process(
        target=_call_paddlex_predict,
        args=(task_path, predict_status_path, params, img_list, img_data,
              save_dir, score_thresh, epoch))
    p.start()
    set_folder_status(predict_status_path, PredictStatus.XPRESTART, p.pid)
    return p, save_dir


def stop_predict_task(task_path):
    """停止预测任务

    Args:
        task_path(str): 从predict_path下的'XPRESTART'文件中获取训练的进程id
    """
    from ..utils import list_files
    predict_status_path = osp.join(task_path, "./logs/predict")
    save_dir = osp.join(task_path, "visualized_test_results")
    if not osp.exists(save_dir):
        return None, "任务目录下没有visualized_test_results文件夹，{}".format(
            task_path), 0, 0
    status, message = get_folder_status(predict_status_path, True)
    if status == PredictStatus.XPRESTART:
        pid = int(message)
        is_dead = False
        if not psutil.pid_exists(pid):
            is_dead = True
        else:
            p = psutil.Process(pid)
            if p.status() == 'zombie':
                is_dead = True
        if is_dead:
            status = PredictStatus.XPREFAIL
            message = "图片预测过程出现异常，请尝试重新预测！"
            set_folder_status(predict_status_path, status, message)
        else:
            pkill(pid)
            status = PredictStatus.XPREFAIL
            message = "图片预测进程已停止！"
            set_folder_status(predict_status_path, status, message)
    if status not in [
            PredictStatus.XPRESTART, PredictStatus.XPREDONE,
            PredictStatus.XPREFAIL
    ]:
        raise ValueError("预测任务状态异常，{}".format(status))
    predict_num = len(list_files(save_dir))
    if predict_num > 0:
        total_num = int(
            open(
                osp.join(predict_status_path, "total_num"), encoding='utf-8')
            .readline().strip())
    else:
        predict_num = 0
        total_num = 0
    return status, message, predict_num, total_num


def get_export_status(task_path):
    """获取导出状态

    Args:
        task_path(str): 从task_path下的'export/XEXPORTING'文件中获取训练的进程id
    Return:
        导出的状态和其他消息.
    """
    export_status_path = osp.join(task_path, './logs/export')
    if not osp.exists(export_status_path):
        return None, "{}任务目录下没有export文件夹".format(task_path)
    status, message = get_folder_status(export_status_path, True)
    if status == TaskStatus.XEXPORTING:
        pid = int(message)
        is_dead = False
        if not psutil.pid_exists(pid):
            is_dead = True
        else:
            p = psutil.Process(pid)
            if p.status() == 'zombie':
                is_dead = True
        if is_dead:
            status = TaskStatus.XEXPORTFAIL
            message = "导出过程出现异常，请尝试重新评估！"
            set_folder_status(export_status_path, status, message)
    if status not in [
            TaskStatus.XEXPORTING, TaskStatus.XEXPORTED, TaskStatus.XEXPORTFAIL
    ]:
        # raise ValueError("获取到的导出状态异常，{}。".format(status))
        return None, "获取到的导出状态异常，{}。".format(status)
    return status, message


def export_quant_model(task_path, save_dir, epoch=None):
    """导出量化模型

    Args:
        task_path(str): 模型训练的路径
        save_dir(str): 导出后的模型保存路径
    """
    output_path = osp.join(task_path, 'output')
    if not osp.exists(osp.join(output_path, 'best_model')):
        raise Exception("未在训练路径{}下发现保存的best_model，导出失败".format(output_path))
    export_status_path = osp.join(task_path, './logs/export')
    safe_clean_folder(export_status_path)

    params_conf_file = osp.join(task_path, 'params.pkl')
    assert osp.exists(
        params_conf_file), "任务无法启动，路径{}下不存在参数配置文件params.pkl".format(task_path)
    with open(params_conf_file, 'rb') as f:
        params = pickle.load(f)
    p = mp.Process(
        target=_call_paddlex_export_quant,
        args=(task_path, params, save_dir, export_status_path, epoch))
    p.start()
    set_folder_status(export_status_path, TaskStatus.XEXPORTING, p.pid)
    set_folder_status(task_path, TaskStatus.XEXPORTING, p.pid)
    return p


def export_noquant_model(task_path, save_dir, epoch=None):
    """导出inference模型

    Args:
        task_path(str): 模型训练的路径
        save_dir(str): 导出后的模型保存路径
    """
    output_path = osp.join(task_path, 'output')
    if not osp.exists(osp.join(output_path, 'best_model')):
        raise Exception("未在训练路径{}下发现保存的best_model，导出失败".format(output_path))
    export_status_path = osp.join(task_path, './logs/export')
    safe_clean_folder(export_status_path)
    p = mp.Process(
        target=_call_paddlex_export_infer,
        args=(task_path, save_dir, export_status_path, epoch))
    p.start()
    set_folder_status(export_status_path, TaskStatus.XEXPORTING, p.pid)
    set_folder_status(task_path, TaskStatus.XEXPORTING, p.pid)
    return p


def opt_lite_model(model_path, save_dir=None, place='arm'):
    p = mp.Process(
        target=_call_paddlelite_export_lite,
        args=(model_path, save_dir, place))
    p.start()
    p.join()


def stop_export_task(task_path):
    """停止导出

    Args:
        task_path(str): 从task_path下的'export/XEXPORTING'文件中获取训练的进程id
    Return:
        the export status and message.
    """
    export_status_path = osp.join(task_path, './logs/export')
    if not osp.exists(export_status_path):
        return None, "{}任务目录下没有export文件夹".format(task_path)
    status, message = get_folder_status(export_status_path, True)
    if status == TaskStatus.XEXPORTING:
        pid = int(message)
        is_dead = False
        if not psutil.pid_exists(pid):
            is_dead = True
        else:
            p = psutil.Process(pid)
            if p.status() == 'zombie':
                is_dead = True
        if is_dead:
            status = TaskStatus.XEXPORTFAIL
            message = "导出过程出现异常，请尝试重新评估！"
            set_folder_status(export_status_path, status, message)
        else:
            pkill(pid)
            status = TaskStatus.XEXPORTFAIL
            message = "已停止导出进程！"
            set_folder_status(export_status_path, status, message)
    if status not in [
            TaskStatus.XEXPORTING, TaskStatus.XEXPORTED, TaskStatus.XEXPORTFAIL
    ]:
        raise ValueError("获取到的导出状态异常，{}。".format(status))
    return status, message
