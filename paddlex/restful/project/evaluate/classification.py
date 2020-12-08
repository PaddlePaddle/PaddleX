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
import os.path as osp
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


class Evaluator(object):
    def __init__(self, model_path, topk=5):
        with open(osp.join(model_path, "model.yml")) as f:
            model_info = yaml.load(f.read(), Loader=yaml.Loader)
        with open(osp.join(model_path, 'eval_details.json'), 'r') as f:
            eval_details = json.load(f)
        self.topk = topk

        self.labels = model_info['_Attributes']['labels']
        self.true_labels = np.array(eval_details['true_labels'])
        self.pred_scores = np.array(eval_details['pred_scores'])
        label_ids_list = list(range(len(self.labels)))
        self.no_appear_label_ids = set(label_ids_list) - set(
            self.true_labels.tolist())

    def cal_confusion_matrix(self):
        '''计算混淆矩阵。
        '''
        pred_labels = np.argsort(self.pred_scores)[:, -1:].flatten()
        cm = confusion_matrix(
            self.true_labels.tolist(),
            pred_labels.tolist(),
            labels=list(range(len(self.labels))))
        return cm

    def cal_precision_recall_F1(self):
        '''计算precision、recall、F1。
        '''
        out = {}
        out_avg = {}
        out_avg['precision'] = 0.0
        out_avg['recall'] = 0.0
        out_avg['F1'] = 0.0
        pred_labels = np.argsort(self.pred_scores)[:, -1:].flatten()
        for label_id in range(len(self.labels)):
            out[self.labels[label_id]] = {}
            if label_id in self.no_appear_label_ids:
                out[self.labels[label_id]]['precision'] = -1.0
                out[self.labels[label_id]]['recall'] = -1.0
                out[self.labels[label_id]]['F1'] = -1.0
                continue
            pred_index = np.where(pred_labels == label_id)[0].tolist()
            tp = np.sum(
                self.true_labels[pred_index] == pred_labels[pred_index])
            tp_fp = len(pred_index)
            tp_fn = len(np.where(self.true_labels == label_id)[0].tolist())
            out[self.labels[label_id]]['precision'] = tp * 1.0 / tp_fp
            out[self.labels[label_id]]['recall'] = tp * 1.0 / tp_fn
            out[self.labels[label_id]]['F1'] = 2 * tp * 1.0 / (tp_fp + tp_fn)
            ratio = tp_fn * 1.0 / self.true_labels.shape[0]
            out_avg['precision'] += out[self.labels[label_id]][
                'precision'] * ratio
            out_avg['recall'] += out[self.labels[label_id]]['recall'] * ratio
            out_avg['F1'] += out[self.labels[label_id]]['F1'] * ratio
        return out, out_avg

    def cal_auc(self):
        '''计算AUC。
        '''
        out = {}
        for label_id in range(len(self.labels)):
            part_pred_scores = self.pred_scores[:, label_id:label_id + 1]
            part_pred_scores = part_pred_scores.flatten()
            fpr, tpr, thresholds = roc_curve(
                self.true_labels, part_pred_scores, pos_label=label_id)
            label_auc = auc(fpr, tpr)
            if label_id in self.no_appear_label_ids:
                out[self.labels[label_id]] = -1.0
                continue
            out[self.labels[label_id]] = label_auc
        return out

    def cal_accuracy(self):
        '''计算Accuracy。
        '''
        out = {}
        k = min(self.topk, len(self.labels))
        pred_top1_label = np.argsort(self.pred_scores)[:, -1]
        pred_topk_label = np.argsort(self.pred_scores)[:, -k:]
        acc1 = sum(pred_top1_label == self.true_labels) / len(self.true_labels)
        acck = sum([
            np.isin(x, y) for x, y in zip(self.true_labels, pred_topk_label)
        ]) / len(self.true_labels)
        out['acc1'] = acc1
        out['acck'] = acck
        out['k'] = k
        return out

    def generate_report(self):
        '''生成评估报告。
        '''
        report = dict()
        report['Confusion_Matrix'] = self.cal_confusion_matrix()
        report['PRF1_average'] = {}
        report['PRF1'], report['PRF1_average'][
            'over_all'] = self.cal_precision_recall_F1()
        auc = self.cal_auc()
        for k, v in auc.items():
            report['PRF1'][k]['auc'] = v
        acc = self.cal_accuracy()
        report["Acc1"] = acc["acc1"]
        report["Acck"] = acc["acck"]
        report["topk"] = acc["k"]
        report['label_list'] = self.labels
        return report
