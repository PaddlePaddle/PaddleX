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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import optimizer as optim

from paddlex.ppcls.utils import logger


class Momentum(object):
    """
    Simple Momentum optimizer with velocity state.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """

    def __init__(self,
                 learning_rate,
                 momentum,
                 weight_decay=None,
                 grad_clip=None,
                 multi_precision=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.multi_precision = multi_precision

    def __call__(self, model_list):
        # model_list is None in static graph
        parameters = sum([m.parameters() for m in model_list],
                         []) if model_list else None
        opt = optim.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            multi_precision=self.multi_precision,
            parameters=parameters)
        return opt


class Adam(object):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 parameter_list=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None,
                 lazy_mode=False,
                 multi_precision=False):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameter_list = parameter_list
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.name = name
        self.lazy_mode = lazy_mode
        self.multi_precision = multi_precision

    def __call__(self, model_list):
        # model_list is None in static graph
        parameters = sum([m.parameters() for m in model_list],
                         []) if model_list else None
        opt = optim.Adam(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            name=self.name,
            lazy_mode=self.lazy_mode,
            multi_precision=self.multi_precision,
            parameters=parameters)
        return opt


class RMSProp(object):
    """
    Root Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning rate method.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        rho (float) - rho value in equation.
        epsilon (float) - avoid division by zero, default is 1e-6.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """

    def __init__(self,
                 learning_rate,
                 momentum=0.0,
                 rho=0.95,
                 epsilon=1e-6,
                 weight_decay=None,
                 grad_clip=None,
                 multi_precision=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model_list):
        # model_list is None in static graph
        parameters = sum([m.parameters() for m in model_list],
                         []) if model_list else None
        opt = optim.RMSProp(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            rho=self.rho,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            parameters=parameters)
        return opt


class AdamW(object):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=None,
                 multi_precision=False,
                 grad_clip=None,
                 no_weight_decay_name=None,
                 one_dim_param_no_weight_decay=False,
                 **args):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.multi_precision = multi_precision
        self.no_weight_decay_name_list = no_weight_decay_name.split(
        ) if no_weight_decay_name else []
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay

    def __call__(self, model_list):
        # model_list is None in static graph
        parameters = sum([m.parameters() for m in model_list],
                         []) if model_list else None

        # TODO(gaotingquan): model_list is None when in static graph, "no_weight_decay" not work.
        if model_list is None:
            if self.one_dim_param_no_weight_decay or len(
                    self.no_weight_decay_name_list) != 0:
                msg = "\"AdamW\" does not support setting \"no_weight_decay\" in static graph. Please use dynamic graph."
                logger.error(Exception(msg))
                raise Exception(msg)

        self.no_weight_decay_param_name_list = [
            p.name for model in model_list for n, p in model.named_parameters()
            if any(nd in n for nd in self.no_weight_decay_name_list)
        ] if model_list else []

        if self.one_dim_param_no_weight_decay:
            self.no_weight_decay_param_name_list += [
                p.name
                for model in model_list for n, p in model.named_parameters()
                if len(p.shape) == 1
            ] if model_list else []

        opt = optim.AdamW(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            parameters=parameters,
            weight_decay=self.weight_decay,
            multi_precision=self.multi_precision,
            grad_clip=self.grad_clip,
            apply_decay_param_fun=self._apply_decay_param_fun)
        return opt

    def _apply_decay_param_fun(self, name):
        return name not in self.no_weight_decay_param_name_list
