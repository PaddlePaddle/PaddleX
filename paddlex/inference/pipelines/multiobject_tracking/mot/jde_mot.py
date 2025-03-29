# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

from collections import defaultdict

from .base_mot import BaseMOT


class JDEMOT(BaseMOT):

    def __init__(self, *args, num_classes, **kwargs):
        self.num_classes = num_classes
        super().__init__(*args, **kwargs)

    def tracking(self, img, **kwargs):
        det_result = next(self.detector(img))
        pred_dets, pred_embs = det_result["pred_dets"], det_result["pred_embs"]
        online_targets_dict = self.tracker.update(pred_dets, pred_embs)

        online_results = defaultdict(list)
        for cls_id in range(self.num_classes):
            online_targets = online_targets_dict[cls_id]
            for t in online_targets:
                tlwh = t.tlwh
                if tlwh[2] * tlwh[3] <= self.tracker.min_box_area:
                    continue
                if (
                    self.tracker.vertical_ratio > 0
                    and tlwh[2] / tlwh[3] > self.tracker.vertical_ratio
                ):
                    continue
                online_results[cls_id].append(t)
        return online_results
