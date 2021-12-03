#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn

from .celoss import CELoss
from .dmlloss import DMLLoss
from .distanceloss import DistanceLoss


class DistillationCELoss(CELoss):
    """
    DistillationCELoss
    """

    def __init__(self,
                 model_name_pairs=[],
                 epsilon=None,
                 key=None,
                 name="loss_ce"):
        super().__init__(epsilon=epsilon)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.name = name

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            loss = super().forward(out1, out2)
            for key in loss:
                loss_dict["{}_{}_{}".format(key, pair[0], pair[1])] = loss[key]
        return loss_dict


class DistillationGTCELoss(CELoss):
    """
    DistillationGTCELoss
    """

    def __init__(self,
                 model_names=[],
                 epsilon=None,
                 key=None,
                 name="loss_gt_ce"):
        super().__init__(epsilon=epsilon)
        assert isinstance(model_names, list)
        self.key = key
        self.model_names = model_names
        self.name = name

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, name in enumerate(self.model_names):
            out = predicts[name]
            if self.key is not None:
                out = out[self.key]
            loss = super().forward(out, batch)
            for key in loss:
                loss_dict["{}_{}".format(key, name)] = loss[key]
        return loss_dict


class DistillationDMLLoss(DMLLoss):
    """
    """

    def __init__(self,
                 model_name_pairs=[],
                 act=None,
                 key=None,
                 name="loss_dml"):
        super().__init__(act=act)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.name = name

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            loss = super().forward(out1, out2)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1],
                                                   idx)] = loss[key]
            else:
                loss_dict["{}_{}".format(self.name, idx)] = loss
        return loss_dict


class DistillationDistanceLoss(DistanceLoss):
    """
    """

    def __init__(self,
                 mode="l2",
                 model_name_pairs=[],
                 key=None,
                 name="loss_",
                 **kargs):
        super().__init__(mode=mode, **kargs)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.name = name + "_l2"

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            loss = super().forward(out1, out2)
            for key in loss:
                loss_dict["{}_{}_{}".format(self.name, key, idx)] = loss[key]
        return loss_dict
