# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import, division, print_function

import time
import paddle
from paddlex.ppcls.engine.train.utils import update_loss, update_metric, log_info
from paddlex.ppcls.utils import profiler


def train_epoch(engine, epoch_id, print_batch_step):
    tic = time.time()
    for iter_id, batch in enumerate(engine.train_dataloader):
        if iter_id >= engine.max_iter:
            break
        profiler.add_profiler_step(engine.config["profiler_options"])
        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        batch_size = batch[0].shape[0]
        if not engine.config["Global"].get("use_multilabel", False):
            batch[1] = batch[1].reshape([batch_size, -1])
        engine.global_step += 1

        # image input
        if engine.amp:
            with paddle.amp.auto_cast(custom_black_list={
                    "flatten_contiguous_range", "greater_than"
            }):
                out = forward(engine, batch)
        else:
            out = forward(engine, batch)

        loss_dict = engine.train_loss_func(out, batch[1])

        # step opt and lr
        if engine.amp:
            scaled = engine.scaler.scale(loss_dict["loss"])
            scaled.backward()
            engine.scaler.minimize(engine.optimizer, scaled)
        else:
            loss_dict["loss"].backward()
            engine.optimizer.step()
        engine.optimizer.clear_grad()
        engine.lr_sch.step()

        # below code just for logging
        # update metric_for_logger
        update_metric(engine, out, batch, batch_size)
        # update_loss_for_logger
        update_loss(engine, loss_dict, batch_size)
        engine.time_info["batch_cost"].update(time.time() - tic)
        if iter_id % print_batch_step == 0:
            log_info(engine, batch_size, epoch_id, iter_id)
        tic = time.time()


def forward(engine, batch):
    if not engine.is_rec:
        return engine.model(batch[0])
    else:
        return engine.model(batch[0], batch[1])
