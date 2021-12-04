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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from paddle.optimizer import lr
from paddle.optimizer.lr import LRScheduler

from paddlex.ppcls.utils import logger


class Linear(object):
    """
    Linear learning rate decay
    Args:
        lr (float): The initial learning rate. It is a python float number.
        epochs(int): The decay step size. It determines the decay cycle.
        end_lr(float, optional): The minimum final learning rate. Default: 0.0001.
        power(float, optional): Power of polynomial. Default: 1.0.
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 epochs,
                 step_each_epoch,
                 end_lr=0.0,
                 power=1.0,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 **kwargs):
        super().__init__()
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"Global.epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logger.warning(msg)
            warmup_epoch = epochs
        self.learning_rate = learning_rate
        self.steps = (epochs - warmup_epoch) * step_each_epoch
        self.end_lr = end_lr
        self.power = power
        self.last_epoch = last_epoch
        self.warmup_steps = round(warmup_epoch * step_each_epoch)
        self.warmup_start_lr = warmup_start_lr

    def __call__(self):
        learning_rate = lr.PolynomialDecay(
            learning_rate=self.learning_rate,
            decay_steps=self.steps,
            end_lr=self.end_lr,
            power=self.power,
            last_epoch=self.
            last_epoch) if self.steps > 0 else self.learning_rate
        if self.warmup_steps > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_steps,
                start_lr=self.warmup_start_lr,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


class Cosine(object):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        eta_min(float): Minimum learning rate. Default: 0.0.
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 eta_min=0.0,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 **kwargs):
        super().__init__()
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"Global.epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logger.warning(msg)
            warmup_epoch = epochs
        self.learning_rate = learning_rate
        self.T_max = (epochs - warmup_epoch) * step_each_epoch
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.warmup_steps = round(warmup_epoch * step_each_epoch)
        self.warmup_start_lr = warmup_start_lr

    def __call__(self):
        learning_rate = lr.CosineAnnealingDecay(
            learning_rate=self.learning_rate,
            T_max=self.T_max,
            eta_min=self.eta_min,
            last_epoch=self.
            last_epoch) if self.T_max > 0 else self.learning_rate
        if self.warmup_steps > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_steps,
                start_lr=self.warmup_start_lr,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


class Step(object):
    """
    Piecewise learning rate decay
    Args:
        step_each_epoch(int): steps each epoch
        learning_rate (float): The initial learning rate. It is a python float number.
        step_size (int): the interval to update.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
            It should be less than 1.0. Default: 0.1.
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 step_size,
                 step_each_epoch,
                 epochs,
                 gamma,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 **kwargs):
        super().__init__()
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"Global.epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logger.warning(msg)
            warmup_epoch = epochs
        self.step_size = step_each_epoch * step_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.warmup_steps = round(warmup_epoch * step_each_epoch)
        self.warmup_start_lr = warmup_start_lr

    def __call__(self):
        learning_rate = lr.StepDecay(
            learning_rate=self.learning_rate,
            step_size=self.step_size,
            gamma=self.gamma,
            last_epoch=self.last_epoch)
        if self.warmup_steps > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_steps,
                start_lr=self.warmup_start_lr,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


class Piecewise(object):
    """
    Piecewise learning rate decay
    Args:
        boundaries(list): A list of steps numbers. The type of element in the list is python int.
        values(list): A list of learning rate values that will be picked during different epoch boundaries.
            The type of element in the list is python float.
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 step_each_epoch,
                 decay_epochs,
                 values,
                 epochs,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 **kwargs):
        super().__init__()
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"Global.epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logger.warning(msg)
            warmup_epoch = epochs
        self.boundaries = [step_each_epoch * e for e in decay_epochs]
        self.values = values
        self.last_epoch = last_epoch
        self.warmup_steps = round(warmup_epoch * step_each_epoch)
        self.warmup_start_lr = warmup_start_lr

    def __call__(self):
        learning_rate = lr.PiecewiseDecay(
            boundaries=self.boundaries,
            values=self.values,
            last_epoch=self.last_epoch)
        if self.warmup_steps > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_steps,
                start_lr=self.warmup_start_lr,
                end_lr=self.values[0],
                last_epoch=self.last_epoch)
        return learning_rate


class MultiStepDecay(LRScheduler):
    """
    Update the learning rate by ``gamma`` once ``epoch`` reaches one of the milestones.
    The algorithm can be described as the code below.
    .. code-block:: text
        learning_rate = 0.5
        milestones = [30, 50]
        gamma = 0.1
        if epoch < 30:
            learning_rate = 0.5
        elif epoch < 50:
            learning_rate = 0.05
        else:
            learning_rate = 0.005
    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        milestones (tuple|list): List or tuple of each boundaries. Must be increasing.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
            It should be less than 1.0. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``MultiStepDecay`` instance to schedule learning rate.
    Examples:

        .. code-block:: python
            import paddle
            import numpy as np
            # train on default dynamic graph mode
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2, 4, 6], gamma=0.8, verbose=True)
            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            for epoch in range(20):
                for batch_id in range(5):
                    x = paddle.uniform([10, 10])
                    out = linear(x)
                    loss = paddle.mean(out)
                    loss.backward()
                    sgd.step()
                    sgd.clear_gradients()
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
            # train on static graph mode
            paddle.enable_static()
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                x = paddle.static.data(name='x', shape=[None, 4, 5])
                y = paddle.static.data(name='y', shape=[None, 4, 5])
                z = paddle.static.nn.fc(x, 100)
                loss = paddle.mean(z)
                scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.5, milestones=[2, 4, 6], gamma=0.8, verbose=True)
                sgd = paddle.optimizer.SGD(learning_rate=scheduler)
                sgd.minimize(loss)
            exe = paddle.static.Executor()
            exe.run(start_prog)
            for epoch in range(20):
                for batch_id in range(5):
                    out = exe.run(
                        main_prog,
                        feed={
                            'x': np.random.randn(3, 4, 5).astype('float32'),
                            'y': np.random.randn(3, 4, 5).astype('float32')
                        },
                        fetch_list=loss.name)
                    scheduler.step()    # If you update learning rate each step
              # scheduler.step()        # If you update learning rate each epoch
    """

    def __init__(self,
                 learning_rate,
                 milestones,
                 epochs,
                 step_each_epoch,
                 gamma=0.1,
                 last_epoch=-1,
                 verbose=False):
        if not isinstance(milestones, (tuple, list)):
            raise TypeError(
                "The type of 'milestones' in 'MultiStepDecay' must be 'tuple, list', but received %s."
                % type(milestones))
        if not all([
                milestones[i] < milestones[i + 1]
                for i in range(len(milestones) - 1)
        ]):
            raise ValueError('The elements of milestones must be incremented')
        if gamma >= 1.0:
            raise ValueError('gamma should be < 1.0.')
        self.milestones = [x * step_each_epoch for x in milestones]
        self.gamma = gamma
        super().__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        for i in range(len(self.milestones)):
            if self.last_epoch < self.milestones[i]:
                return self.base_lr * (self.gamma**i)
        return self.base_lr * (self.gamma**len(self.milestones))
