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

import json
import yaml
import os.path as osp
import numpy as np


class Evaluator(object):
    def __init__(self, model_path):
        with open(osp.join(model_path, "model.yml")) as f:
            model_info = yaml.load(f.read(), Loader=yaml.Loader)
        self.labels = model_info['_Attributes']['labels']
        with open(osp.join(model_path, 'eval_details.json'), 'r') as f:
            eval_details = json.load(f)
        self.confusion_matrix = np.array(eval_details['confusion_matrix'])
        self.num_classes = len(self.confusion_matrix)

    def cal_iou(self):
        '''计算IoU。
        '''
        category_iou = []
        mean_iou = 0
        vji = np.sum(self.confusion_matrix, axis=1)
        vij = np.sum(self.confusion_matrix, axis=0)

        for c in range(self.num_classes):
            total = vji[c] + vij[c] - self.confusion_matrix[c][c]
            if total == 0:
                iou = 0
            else:
                iou = float(self.confusion_matrix[c][c]) / total
            mean_iou += iou
            category_iou.append(iou)
        mean_iou = float(mean_iou) / float(self.num_classes)
        return np.array(category_iou), mean_iou

    def cal_acc(self):
        '''计算Acc。
        '''
        total = self.confusion_matrix.sum()
        total_tp = 0
        for c in range(self.num_classes):
            total_tp += self.confusion_matrix[c][c]
        if total == 0:
            mean_acc = 0
        else:
            mean_acc = float(total_tp) / total

        vij = np.sum(self.confusion_matrix, axis=0)
        category_acc = []
        for c in range(self.num_classes):
            if vij[c] == 0:
                acc = 0
            else:
                acc = self.confusion_matrix[c][c] / float(vij[c])
            category_acc.append(acc)
        return np.array(category_acc), mean_acc

    def cal_confusion_matrix(self):
        '''计算混淆矩阵。
        '''
        return self.confusion_matrix

    def cal_precision_recall(self):
        '''计算precision、recall.
        '''
        self.precision_recall = dict()
        for i in range(len(self.labels)):
            label_name = self.labels[i]
            if np.isclose(np.sum(self.confusion_matrix[i, :]), 0, atol=1e-6):
                recall = -1
            else:
                total_gt = np.sum(self.confusion_matrix[i, :]) + 1e-06
                recall = self.confusion_matrix[i, i] / total_gt
            if np.isclose(np.sum(self.confusion_matrix[:, i]), 0, atol=1e-6):
                precision = -1
            else:
                total_pred = np.sum(self.confusion_matrix[:, i]) + 1e-06
                precision = self.confusion_matrix[i, i] / total_pred
            self.precision_recall[label_name] = {
                'precision': precision,
                'recall': recall
            }
        return self.precision_recall

    def generate_report(self):
        '''生成评估报告。
        '''
        category_iou, mean_iou = self.cal_iou()
        category_acc, mean_acc = self.cal_acc()

        category_iou_dict = {}
        for i in range(len(category_iou)):
            category_iou_dict[self.labels[i]] = category_iou[i]

        report = dict()
        report['Confusion_Matrix'] = self.cal_confusion_matrix()
        report['Mean_IoU'] = mean_iou
        report['Mean_Acc'] = mean_acc
        report['PRIoU'] = self.cal_precision_recall()
        for key in report['PRIoU']:
            report['PRIoU'][key]["iou"] = category_iou_dict[key]
        report['label_list'] = self.labels
        return report
