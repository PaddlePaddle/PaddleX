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

import copy
import importlib

import paddle.nn as nn
from paddle.jit import to_static
from paddle.static import InputSpec

from . import backbone, gears
from .backbone import *
from .gears import build_gear
from .utils import *
from paddlex.ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from paddlex.ppcls.utils import logger
from paddlex.ppcls.utils.save_load import load_dygraph_pretrain

__all__ = ["build_model", "RecModel", "DistillationModel"]


def build_model(config):
    config = copy.deepcopy(config)
    model_type = config.pop("name")
    mod = importlib.import_module(__name__)
    arch = getattr(mod, model_type)(**config)
    return arch


def apply_to_static(config, model):
    support_to_static = config['Global'].get('to_static', False)

    if support_to_static:
        specs = None
        if 'image_shape' in config['Global']:
            specs = [InputSpec([None] + config['Global']['image_shape'])]
        model = to_static(model, input_spec=specs)
        logger.info("Successfully to apply @to_static with specs: {}".format(
            specs))
    return model


class RecModel(nn.Layer):
    def __init__(self, **config):
        super().__init__()
        backbone_config = config["Backbone"]
        backbone_name = backbone_config.pop("name")
        self.backbone = eval(backbone_name)(**backbone_config)
        if "BackboneStopLayer" in config:
            backbone_stop_layer = config["BackboneStopLayer"]["name"]
            self.backbone.stop_after(backbone_stop_layer)

        if "Neck" in config:
            self.neck = build_gear(config["Neck"])
        else:
            self.neck = None

        if "Head" in config:
            self.head = build_gear(config["Head"])
        else:
            self.head = None

    def forward(self, x, label=None):
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        if self.head is not None:
            y = self.head(x, label)
        else:
            y = None
        return {"features": x, "logits": y}


class DistillationModel(nn.Layer):
    def __init__(self,
                 models=None,
                 pretrained_list=None,
                 freeze_params_list=None,
                 **kargs):
        super().__init__()
        assert isinstance(models, list)
        self.model_list = []
        self.model_name_list = []
        if pretrained_list is not None:
            assert len(pretrained_list) == len(models)

        if freeze_params_list is None:
            freeze_params_list = [False] * len(models)
        assert len(freeze_params_list) == len(models)
        for idx, model_config in enumerate(models):
            assert len(model_config) == 1
            key = list(model_config.keys())[0]
            model_config = model_config[key]
            model_name = model_config.pop("name")
            model = eval(model_name)(**model_config)

            if freeze_params_list[idx]:
                for param in model.parameters():
                    param.trainable = False
            self.model_list.append(self.add_sublayer(key, model))
            self.model_name_list.append(key)

        if pretrained_list is not None:
            for idx, pretrained in enumerate(pretrained_list):
                if pretrained is not None:
                    load_dygraph_pretrain(
                        self.model_name_list[idx], path=pretrained)

    def forward(self, x, label=None):
        result_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            if label is None:
                result_dict[model_name] = self.model_list[idx](x)
            else:
                result_dict[model_name] = self.model_list[idx](x, label)
        return result_dict
