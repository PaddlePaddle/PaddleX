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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform
import paddle
from paddlex.ppcls.utils import logger


def retrieval_eval(engine, epoch_id=0):
    engine.model.eval()
    # step1. build gallery
    if engine.gallery_query_dataloader is not None:
        gallery_feas, gallery_img_id, gallery_unique_id = cal_feature(
            engine, name='gallery_query')
        query_feas, query_img_id, query_query_id = gallery_feas, gallery_img_id, gallery_unique_id
    else:
        gallery_feas, gallery_img_id, gallery_unique_id = cal_feature(
            engine, name='gallery')
        query_feas, query_img_id, query_query_id = cal_feature(
            engine, name='query')

    # step2. do evaluation
    sim_block_size = engine.config["Global"].get("sim_block_size", 64)
    sections = [sim_block_size] * (len(query_feas) // sim_block_size)
    if len(query_feas) % sim_block_size:
        sections.append(len(query_feas) % sim_block_size)
    fea_blocks = paddle.split(query_feas, num_or_sections=sections)
    if query_query_id is not None:
        query_id_blocks = paddle.split(
            query_query_id, num_or_sections=sections)
    image_id_blocks = paddle.split(query_img_id, num_or_sections=sections)
    metric_key = None

    if engine.eval_loss_func is None:
        metric_dict = {metric_key: 0.}
    else:
        metric_dict = dict()
        for block_idx, block_fea in enumerate(fea_blocks):
            similarity_matrix = paddle.matmul(
                block_fea, gallery_feas, transpose_y=True)
            if query_query_id is not None:
                query_id_block = query_id_blocks[block_idx]
                query_id_mask = (query_id_block != gallery_unique_id.t())

                image_id_block = image_id_blocks[block_idx]
                image_id_mask = (image_id_block != gallery_img_id.t())

                keep_mask = paddle.logical_or(query_id_mask, image_id_mask)
                similarity_matrix = similarity_matrix * keep_mask.astype(
                    "float32")
            else:
                keep_mask = None

            metric_tmp = engine.eval_metric_func(similarity_matrix,
                                                 image_id_blocks[block_idx],
                                                 gallery_img_id, keep_mask)

            for key in metric_tmp:
                if key not in metric_dict:
                    metric_dict[key] = metric_tmp[key] * block_fea.shape[
                        0] / len(query_feas)
                else:
                    metric_dict[key] += metric_tmp[key] * block_fea.shape[
                        0] / len(query_feas)

    metric_info_list = []
    for key in metric_dict:
        if metric_key is None:
            metric_key = key
        metric_info_list.append("{}: {:.5f}".format(key, metric_dict[key]))
    metric_msg = ", ".join(metric_info_list)
    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

    return metric_dict[metric_key]


def cal_feature(engine, name='gallery'):
    all_feas = None
    all_image_id = None
    all_unique_id = None
    has_unique_id = False

    if name == 'gallery':
        dataloader = engine.gallery_dataloader
    elif name == 'query':
        dataloader = engine.query_dataloader
    elif name == 'gallery_query':
        dataloader = engine.gallery_query_dataloader
    else:
        raise RuntimeError("Only support gallery or query dataset")

    max_iter = len(dataloader) - 1 if platform.system() == "Windows" else len(
        dataloader)
    for idx, batch in enumerate(dataloader):  # load is very time-consuming
        if idx >= max_iter:
            break
        if idx % engine.config["Global"]["print_batch_step"] == 0:
            logger.info(
                f"{name} feature calculation process: [{idx}/{len(dataloader)}]"
            )
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        batch = [paddle.to_tensor(x) for x in batch]
        batch[1] = batch[1].reshape([-1, 1]).astype("int64")
        if len(batch) == 3:
            has_unique_id = True
            batch[2] = batch[2].reshape([-1, 1]).astype("int64")
        out = engine.model(batch[0], batch[1])
        batch_feas = out["features"]

        # do norm
        if engine.config["Global"].get("feature_normalize", True):
            feas_norm = paddle.sqrt(
                paddle.sum(paddle.square(batch_feas), axis=1, keepdim=True))
            batch_feas = paddle.divide(batch_feas, feas_norm)

        # do binarize
        if engine.config["Global"].get("feature_binarize") == "round":
            batch_feas = paddle.round(batch_feas).astype("float32") * 2.0 - 1.0

        if engine.config["Global"].get("feature_binarize") == "sign":
            batch_feas = paddle.sign(batch_feas).astype("float32")

        if all_feas is None:
            all_feas = batch_feas
            if has_unique_id:
                all_unique_id = batch[2]
            all_image_id = batch[1]
        else:
            all_feas = paddle.concat([all_feas, batch_feas])
            all_image_id = paddle.concat([all_image_id, batch[1]])
            if has_unique_id:
                all_unique_id = paddle.concat([all_unique_id, batch[2]])

    if engine.use_dali:
        dataloader.reset()

    if paddle.distributed.get_world_size() > 1:
        feat_list = []
        img_id_list = []
        unique_id_list = []
        paddle.distributed.all_gather(feat_list, all_feas)
        paddle.distributed.all_gather(img_id_list, all_image_id)
        all_feas = paddle.concat(feat_list, axis=0)
        all_image_id = paddle.concat(img_id_list, axis=0)
        if has_unique_id:
            paddle.distributed.all_gather(unique_id_list, all_unique_id)
            all_unique_id = paddle.concat(unique_id_list, axis=0)

    logger.info("Build {} done, all feat shape: {}, begin to eval..".format(
        name, all_feas.shape))
    return all_feas, all_image_id, all_unique_id
