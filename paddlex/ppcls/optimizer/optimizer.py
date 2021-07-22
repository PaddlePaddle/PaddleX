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

import sys

import paddle
import paddle.regularizer as regularizer

__all__ = ['OptimizerBuilder']


class L1Decay(object):
    """
    L1 Weight Decay Regularization, which encourages the weights to be sparse.

    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super(L1Decay, self).__init__()
        self.factor = factor

    def __call__(self):
        reg = regularizer.L1Decay(self.factor)
        return reg


class L2Decay(object):
    """
    L2 Weight Decay Regularization, which encourages the weights to be sparse.

    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super(L2Decay, self).__init__()
        self.factor = factor

    def __call__(self):
        reg = regularizer.L2Decay(self.factor)
        return reg


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
                 parameter_list=None,
                 regularization=None,
                 multi_precision=False,
                 **args):
        super(Momentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.parameter_list = parameter_list
        self.regularization = regularization
        self.multi_precision = multi_precision

    def __call__(self):
        opt = paddle.optimizer.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            parameters=self.parameter_list,
            weight_decay=self.regularization,
            multi_precision=self.multi_precision)
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
                 momentum,
                 rho=0.95,
                 epsilon=1e-6,
                 parameter_list=None,
                 regularization=None,
                 **args):
        super(RMSProp, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon
        self.parameter_list = parameter_list
        self.regularization = regularization

    def __call__(self):
        opt = paddle.optimizer.RMSProp(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            rho=self.rho,
            epsilon=self.epsilon,
            parameters=self.parameter_list,
            weight_decay=self.regularization)
        return opt


class OptimizerBuilder(object):
    """
    Build optimizer

    Args:
        function(str): optimizer name of learning rate
        params(dict): parameters used for init the class
        regularizer (dict): parameters used for create regularization
    """

    def __init__(self,
                 function='Momentum',
                 params={'momentum': 0.9},
                 regularizer=None):
        self.function = function
        self.params = params
        # create regularizer
        if regularizer is not None:
            mod = sys.modules[__name__]
            reg_func = regularizer['function'] + 'Decay'
            del regularizer['function']
            reg = getattr(mod, reg_func)(**regularizer)()
            self.params['regularization'] = reg

    def __call__(self, learning_rate, parameter_list=None):
        mod = sys.modules[__name__]
        opt = getattr(mod, self.function)
        return opt(learning_rate=learning_rate,
                   parameter_list=parameter_list,
                   **self.params)()
